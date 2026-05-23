# Phân Tích Chi Tiết Codebase `mjlab`

---

## 1. Tổng Quan — mjlab Là Gì?

**mjlab** là framework GPU-accelerated robot learning, kết hợp:
- **MuJoCo Warp** (Google DeepMind) — physics engine chạy song song trên GPU với NVIDIA CUDA/Warp
- **Isaac Lab API** — pattern thiết kế Manager-based MDP
- **RSL-RL** — thư viện PPO on-policy training đã được chứng minh trên legged robots

**Đặc điểm cốt lõi:**
- Chạy **4096+ environments song song** trên GPU (một lệnh training duy nhất)
- Thiết kế composable: thay robot, thay terrain, thay reward bằng cách thay config
- Production-ready: ONNX export, W&B logging, multi-GPU, checkpoint migration

---

## 2. Kiến Trúc 7 Tầng

```
[Tầng 7] tasks/          — Task specification (velocity, tracking, cartpole)
[Tầng 6] rl/             — PPO runner, VecEnv wrapper, ONNX export
[Tầng 5] envs/           — ManagerBasedRlEnv — vòng lặp RL chính
[Tầng 4] managers/       — Obs / Action / Reward / Event / Curriculum managers
[Tầng 3] entity/         — Robot model; actuator/ sensor/ terrains/
[Tầng 2] scene/          — World: ghép terrain + robot + sensors vào MjSpec
[Tầng 1] sim/            — GPU physics: MuJoCo Warp + CUDA Graph
```

Mỗi tầng chỉ phụ thuộc vào tầng thấp hơn — không có circular dependency.

---

## 3. Tầng 1 — `sim/` (Physics Engine)

**File chính:** `sim/sim.py`

### Hai dạng model tồn tại song song

| Thuộc tính | Type | Vai trò |
|-----------|------|---------|
| `sim.mj_model` | `mujoco.MjModel` | CPU model — read-only reference |
| `sim.wp_model` | `mjwarp.Model` | GPU Warp model — dùng để simulate |
| `sim.wp_data` | `mjwarp.Data` | GPU data — state của tất cả N environments |

`data.nworld = num_envs` — kích thước của batch GPU.

### 4 phép toán physics cốt lõi

```python
sim.step()     # mj_step: integrate dynamics → cập nhật qpos, qvel
sim.forward()  # mj_forward: recompute derived quantities (xpos, xquat, sensordata)
sim.reset()    # reset_data: đặt env về trạng thái mặc định theo mask
sim.sense()    # Chạy GPU kernels: BVH refit → raycast → camera render
```

**Vì sao `forward()` tách khỏi `step()`?**  
`mj_step` chạy `forward` từ **trước** khi integrate — nên sau khi step xong, `xpos/xquat` lag 1 substep. Framework gọi `forward()` thêm một lần sau khi xử lý reset để sync tất cả envs cùng lúc (thay vì gọi 2 lần).

### CUDA Graph Capture

```python
# Lần đầu: ghi lại chuỗi GPU kernels vào graph
with wp.ScopedCapture() as capture:
    mjwarp.step(model, data)
self.step_graph = capture.graph

# Sau đó: mỗi bước chỉ cần 1 kernel launch thay vì hàng chục
wp.capture_launch(self.step_graph)
```

Giúp giảm CPU overhead đáng kể khi chạy hàng nghìn environments.

### Domain Randomization — `expand_model_fields()`

Mặc định, `wp_model` có **một giá trị dùng chung** cho tất cả environments (e.g., `geom_friction` là scalar). Khi cần DR per-environment, gọi:

```python
sim.expand_model_fields(("geom_friction", "body_mass", ...))
# → allocate tensor shape (num_envs, ...) thay vì scalar
# → phải re-capture CUDA graph vì memory address thay đổi
```

---

## 4. Tầng 2 — `scene/` (World Management)

**File chính:** `scene/scene.py`

`Scene` là "tổng đạo diễn" — ghép tất cả thành phần vào một `mujoco.MjSpec` trước khi compile:

```
SceneCfg
 ├── terrain: TerrainEntityCfg   → _add_terrain() → spec.attach(terrain)
 ├── entities: {name: EntityCfg} → _add_entities() → spec.attach(entity, prefix=name+"/")
 └── sensors: (SensorCfg, ...)   → _add_sensors() → sensor.edit_spec(spec)
```

**Prefix naming:** Mỗi entity được attach với prefix `"robot/"` nên trong XML, joint tên `"left_knee"` trở thành `"robot/left_knee"` — tránh name collision khi có nhiều robots.

**Keyframe:** Nếu entity có `<key>` trong XML (default pose), Scene merge tất cả keyframes thành một `"init_state"` keyframe chung để reset.

---

## 5. Tầng 3a — `actuator/` (Motor Models)

### Phân loại actuators

| Class | Vai trò | Công thức |
|-------|---------|-----------|
| `IdealPdActuator` | PD controller lý tưởng | `τ = kp*(q_des-q) + kd*(qd_des-qd) + τ_ff` |
| `DcMotorActuator` | DC motor với giới hạn tốc độ | `τ_clipped` theo back-EMF |
| `LearnedMlpActuator` | Neural net actuator | MLP(pos_error_history, vel_history) → τ |
| `BuiltinActuator` | Dùng MuJoCo's native actuator | — |
| `XmlActuator` | Actuator định nghĩa trong XML | — |

### `IdealPdActuator` — được dùng nhiều nhất

```python
def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    pos_error = cmd.position_target - cmd.pos   # shape: (num_envs, num_joints)
    vel_error = cmd.velocity_target - cmd.vel
    torques = self.stiffness * pos_error + self.damping * vel_error + cmd.effort_target
    return torch.clamp(torques, -self.force_limit, self.force_limit)
```

`stiffness` và `damping` là tensors `(num_envs, num_joints)` — cho phép DR per-env per-joint.

### `LearnedMlpActuator` — actuator cao cấp

Load TorchScript model offline-trained, dùng **circular buffer** lưu `history_length=3` bước qua khứ của `pos_error` và `vel`, concat thành input cho MLP. Mục đích: capture actuator dynamics phi tuyến (hysteresis, delay, friction) mà analytical model không model được.

### Command Delay — Sim-to-Real Gap

```python
# Trong ActuatorCfg:
delay_min_lag: int = 0    # min latency (physics steps)
delay_max_lag: int = 0    # max latency — set > 0 để enable
```

Model communication latency giữa policy và motor controller. Cực kỳ quan trọng để policy robust khi deploy thực tế.

---

## 6. Tầng 3b — `sensor/`

| Sensor | File | Dùng để |
|--------|------|---------|
| `BuiltinSensor` | `builtin_sensor.py` | IMU (lin_vel, ang_vel), joint encoders, root state |
| `ContactSensor` | `contact_sensor.py` | Foot contact forces, air_time tracking |
| `RayCastSensor` | `raycast_sensor.py` | Height scan — terrain map dưới chân robot |
| `CameraSensor` | `camera_sensor.py` | RGB image cho visual RL |
| `TerrainHeightSensor` | `terrain_height_sensor.py` | Terrain height tại vị trí foot |

**`SensorContext`** — shared GPU resources cho raycast và camera, được capture vào `sense_graph`:
```python
mjwarp.refit_bvh(model, data, rc)   # Rebuild BVH tree (bodies đã di chuyển)
mjwarp.render(model, data, rc)       # Camera rendering
sensor.raycast_kernel(rc=rc)         # Raycast height scan
```

**`ContactSensor`** tracking `air_time`: đếm thời gian từ lúc chân lift-off đến khi tiếp đất → dùng để tính reward khuyến khích gait đẹp.

---

## 7. Tầng 3c — `terrains/`

Terrain được sinh **procedurally** và tile theo grid `num_envs × spacing`:

```python
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    sub_terrains={
        "flat":     FlatTerrainCfg(proportion=0.2),
        "slope":    HfSlopedTerrainCfg(proportion=0.2),
        "stairs":   HfStairTerrainCfg(proportion=0.3),
        "discrete": HfDiscreteObstaclesTerrainCfg(proportion=0.3),
    }
)
```

**Curriculum Terrain:** `env_origins` được sắp xếp từ dễ (flat) đến khó (discrete obstacles). Khi robot thành công trên terrain hiện tại, `CurriculumManager` chuyển nó lên terrain khó hơn.

---

## 8. Tầng 4 — `managers/` (Trái Tim MDP)

Pattern: mỗi manager nhận một `dict[str, TermCfg]` — mỗi entry là một "term" với `func` callable và `params`.

### `ObservationManager` — Pipeline xử lý observation

```
func(env, **params)    # raw tensor (num_envs, dim)
→ noise.apply(obs)     # thêm noise nếu enable_corruption=True
→ obs.clip_(min, max)  # clip tùy chọn
→ obs.mul_(scale)      # scale tùy chọn
→ delay_buffer         # model sensor latency
→ circular_buffer      # stack history cho RNN
→ torch.cat(all_terms) # concatenate thành vector
```

**Tại sao Actor và Critic có obs khác nhau?**
- `actor`: có noise (`enable_corruption=True`) → robust training
- `critic`: không noise, thêm privileged info (contact forces, true foot height) → học value function chính xác hơn
- Đây là kỹ thuật **Privileged Critic** / **Asymmetric AC**, rất phổ biến trong legged robot RL

### `EventManager` — Domain Randomization

4 modes:
| Mode | Khi nào trigger | Ví dụ |
|------|-----------------|-------|
| `startup` | 1 lần khi env init | DR friction, encoder bias, CoM |
| `reset` | Mỗi episode reset | Reset robot pose, joint noise |
| `interval` | Ngẫu nhiên trong episode [1-3s] | Push robot |
| `step` | Mỗi env step | Force lifetime management |

**`@requires_model_fields` decorator** — đánh dấu DR function cần expand field:
```python
@requires_model_fields("geom_friction", recompute=RecomputeLevel.none)
def geom_friction(env, env_ids, asset_cfg, ranges, ...):
    # Randomize friction per-env trong range
    ...
```
EventManager collect tất cả fields này và truyền cho `sim.expand_model_fields()`.

### `RewardManager` — Weighted Sum

```python
def compute(self, dt: float) -> torch.Tensor:
    reward_buf = 0
    for name, cfg in terms:
        value = cfg.func(env, **cfg.params)    # shape: (num_envs,)
        value = value * cfg.weight * dt         # scale by dt để normalize
        value = nan_to_num(value)               # guard NaN từ corrupt physics
        reward_buf += value
    return reward_buf
```

**Tại sao scale by `dt`?** Để tổng reward trong episode không phụ thuộc vào control frequency — cùng behavior cho 50Hz và 100Hz policy.

---

## 9. Tầng 5 — `envs/` (Vòng Lặp RL Chính)

**File chính:** `envs/manager_based_rl_env.py`

### Vòng lặp `step()` — thứ tự quan trọng

```python
def step(action):
    # 1. Scale action → position targets
    action_manager.process_action(action)

    # 2. Decimation loop (decimation=4 steps physics / 1 step policy)
    for _ in range(4):                       # 4 × 0.005s = 0.02s = 50Hz
        action_manager.apply_action()         # targets → actuator → ctrl
        scene.write_data_to_sim()             # write entity data to GPU
        sim.step()                            # MuJoCo Warp GPU step
        scene.update(dt=0.005)                # ContactSensor air_time tracking
        metrics_manager.compute_substep()

    # 3. RL signals — TRƯỚC khi forward()
    termination_manager.compute()             # check fallen, timeout, bounds
    reward_manager.compute(dt=0.02)          # weighted sum rewards

    # 4. Auto-reset terminated envs
    reset_env_ids = reset_buf.nonzero()
    if len(reset_env_ids) > 0:
        _reset_idx(reset_env_ids)             # curriculum → sim.reset → events
        scene.write_data_to_sim()

    # 5. SINGLE forward() call cho TẤT CẢ envs
    sim.forward()                             # sync xpos/xquat từ qpos hiện tại
    # Non-reset envs: resolve 1-substep staleness
    # Reset envs: pick up fresh reset state

    # 6. Sau forward: events, sense, obs
    event_manager.apply(mode="step")
    event_manager.apply(mode="interval")
    sim.sense()                               # raycast, camera
    obs_buf = observation_manager.compute()

    return obs, reward, terminated, truncated, extras
```

### Tại sao `sim.forward()` chỉ gọi 1 lần?

Đây là **design decision quan trọng**: thay vì gọi 2 lần (1 sau decimation loop + 1 sau reset), chỉ gọi 1 lần sau khi reset xong. Lợi ích: tránh double-compute, cả non-reset envs và reset envs đều sync đúng. Trade-off: reward/termination managers thấy derived quantities lag 1 substep — nhưng consistent và policy có thể học được.

### `decimation=4` — ý nghĩa

```
Physics: 0.005s timestep = 200Hz
Policy: 0.005 × 4 = 0.02s = 50Hz control frequency
```

Policy chạy chậm hơn physics để actions có thời gian "settle" — giống robot thực tế.

---

## 10. Tầng 6 — `rl/` (PPO Training)

### Thuật toán: PPO (Proximal Policy Optimization)

Được implement bởi `rsl-rl-lib==5.2.0`, mjlab chỉ wrap và extend.

**Architecture:**
- Actor: MLP `obs_dim → [128, 128, 128] → action_dim` + ELU activation + Gaussian distribution
- Critic: MLP `critic_obs_dim → [128, 128, 128] → 1` (value estimate)

**PPO Loop:**

```
COLLECTION (24 steps × 4096 envs = 98,304 transitions/update):
  obs → actor → action (stochastic, Gaussian)
  env.step(action) → (next_obs, reward, done)

ADVANTAGE ESTIMATION (GAE):
  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
  A_t = Σ_{k=0}^{T} (γλ)^k · δ_{t+k}
  (γ=0.99, λ=0.95)

UPDATE (5 epochs × 4 mini-batches):
  ratio = π_new(a|s) / π_old(a|s)
  L_clip = -E[min(ratio·A, clip(ratio, 1±0.2)·A)]
  L_value = (V_pred - V_target)²
  L_entropy = -H(π)  (entropy bonus, coef=0.005)
  L_total = L_clip + 1.0·L_value - 0.005·L_entropy
```

**Adaptive Learning Rate:** Sau mỗi update, tính KL divergence giữa old và new policy. Nếu KL > `desired_kl=0.01`, giảm LR; nếu KL < 0.5×desired_kl, tăng LR.

### `RslRlVecEnvWrapper` — Glue Layer

Wrap `ManagerBasedRlEnv` thành interface RSL-RL expects:
- `obs_dict["actor"]` → actor observation
- `obs_dict["critic"]` → critic observation  
- `terminated | truncated` → `dones`
- `truncated` → `extras["time_outs"]` (để RSL-RL bootstrap value đúng với infinite horizon)

### `MjlabOnPolicyRunner` — Extensions

Extends RSL-RL's `OnPolicyRunner`:
1. **`save()`**: lưu `.pt` + auto-export `.onnx` + upload W&B
2. **`load()`**: hỗ trợ migrate checkpoint từ các format cũ (rsl-rl 3.x → 4.x → 5.x)
3. **ONNX export** với `dynamo=False` — tránh deprecation warning

---

## 11. Tầng 7 — `tasks/` (Task Specification)

### Task 1: Velocity Tracking

**Mục tiêu:** G1 humanoid theo velocity command `(vx, vy, ωz)`

**Observation vector (actor) — ~220 dims:**
| Term | Dim | Noise |
|------|-----|-------|
| IMU linear velocity | 3 | ±0.5 |
| IMU angular velocity | 3 | ±0.2 |
| Projected gravity | 3 | ±0.05 |
| Joint positions (rel to default) | N | ±0.01 rad |
| Joint velocities | N | ±1.5 rad/s |
| Last action | N | — |
| Command (vx, vy, ωz) | 3 | — |
| Height scan (terrain map) | 187 | ±0.1 |

**Reward shaping:**
```
+2.0 × track_linear_velocity     # exp(-||v_cmd - v_actual||² / 0.25)
+2.0 × track_angular_velocity    # exp(-|ω_cmd - ω_actual|² / 0.5)
+1.0 × upright                   # khuyến khích torso thẳng đứng
+1.0 × pose                      # khuyến khích default pose
-1.0 × joint_pos_limits          # phạt khi vượt giới hạn khớp
-0.1 × action_rate_l2            # phạt thay đổi action đột ngột
-2.0 × foot_clearance            # phạt khi chân không lift đủ cao
-0.25 × foot_swing_height        # phạt swing height sai
-0.1 × foot_slip                 # phạt khi chân trượt
```

**Curriculum:**
```python
# Terrain: tăng difficulty khi robot vượt qua terrain hiện tại
terrain_levels: CurriculumTermCfg(func=mdp.terrain_levels_vel)

# Velocity: tăng dần command range theo số bước train
velocity_stages = [
    {"step": 0,         "lin_vel_x": (-1.0, 1.0)},
    {"step": 120_000,   "lin_vel_x": (-1.5, 2.0)},   # iter 5000
    {"step": 240_000,   "lin_vel_x": (-2.0, 3.0)},   # iter 10000
]
```

### Task 2: Motion Tracking (Imitation Learning)

**Mục tiêu:** Imitate reference motions (từ real robot capture hoặc retargeting)

Thêm `MotionCommandCfg` — load `.npz` file chứa reference trajectories, sample ngẫu nhiên frame trong episode, reward là tracking error so với reference pose/velocity.

---

## 12. Domain Randomization — Sim-to-Real Gap

| DR Term | Mode | Đối tượng | Range | Ý nghĩa |
|---------|------|----------|-------|---------|
| `geom_friction` | startup | foot geoms | [0.3, 1.2] | Biến động bề mặt sàn |
| `encoder_bias` | startup | all joints | ±0.015 rad | Offset cảm biến encoder |
| `body_com_offset` | startup | torso body | ±2.5cm xyz | Lệch tâm khối lượng |
| `push_robot` | interval | root state | ±0.5 m/s | Va chạm bất ngờ |

**Tại sao `startup` thay vì `reset`?** Startup randomize một lần, rồi **fix suốt training** — policy phải học cách generalize, không thể "nhớ" DR value của từng episode. Sau đủ iterations, policy trở nên robust với distribution rộng của parameters.

---

## 13. Entry Points & Tooling

```bash
# Training (single GPU)
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096

# Training (multi-GPU, dùng torchrunx)
uv run train Mjlab-Velocity-Flat-Unitree-G1 --gpu-ids "[0,1]" --env.scene.num-envs 4096

# Evaluation (fetch checkpoint từ W&B)
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path org/mjlab/run-id

# List tất cả tasks đã đăng ký
uv run list-envs

# Debug NaN trong training
uv run viz-nan
```

**ONNX Export Pipeline:**
```
Save checkpoint (.pt)
→ export_policy_to_onnx() [dynamo=False, opset 18]
→ attach_metadata_to_onnx() [obs_dim, action_dim, robot info]
→ wandb.save(onnx_path)
→ Deploy lên robot thực mà không cần PyTorch
```

---

## 14. Câu Hỏi Interview — Chuẩn Bị

| Câu hỏi | Câu trả lời cốt lõi |
|---------|---------------------|
| Tại sao PPO thay vì SAC? | On-policy, stable với massive parallel envs, RSL-RL battle-tested trên legged robots |
| `decimation=4` nghĩa gì? | Policy 50Hz, physics 200Hz — tách control freq khỏi physics stability |
| Actor vs Critic obs khác nhau? | Privileged Critic: critic thấy clean privileged info → better value estimate; actor có noise → robust policy |
| DR giúp gì? | Thu hẹp sim-to-real gap bằng cách train trên distribution parameters thay vì một giá trị fixed |
| Curriculum learning hoạt động? | Terrain difficulty + command velocity tăng dần theo performance và số bước train |
| `sim.forward()` sau reset? | MuJoCo cần recompute `xpos/xquat` từ `qpos` hiện tại; gọi 1 lần sau reset cho tất cả envs |
| CUDA Graph Capture là gì? | Record GPU kernel sequence → replay với 1 launch → giảm CPU overhead dramatically |
| `expand_model_fields()`? | Biến model scalar field (shared across envs) thành per-env tensor để enable per-env DR |
| Learned Actuator dùng để làm gì? | Model complex motor dynamics (delay, nonlinearity, friction) mà PD model không capture được |
| `RslRlVecEnvWrapper` làm gì? | Adapter layer giữa mjlab env interface và rsl_rl VecEnv interface |

---

## 15. Sơ Đồ Luồng Dữ Liệu Hoàn Chỉnh

```
train.py
│
├── load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
│   └── velocity_env_cfg.py → ManagerBasedRlEnvCfg
│
├── ManagerBasedRlEnv(cfg, device)
│   ├── Scene(cfg.scene)
│   │   ├── TerrainEntity  ─────────────────────────┐
│   │   ├── Entity("robot") + EntityCfg              │
│   │   │   └── IdealPdActuator(kp, kd)             │
│   │   └── Sensors(terrain_scan, foot_contact)      │
│   │                                                │
│   ├── Simulation(spec, num_envs)                   │
│   │   ├── mj_model (CPU) ──────────────────────────┘
│   │   ├── wp_model (GPU) ← expand_model_fields(DR fields)
│   │   ├── wp_data  (GPU) ← nworld=4096
│   │   └── CUDA Graphs: step/forward/reset/sense
│   │
│   └── load_managers()
│       ├── EventManager  ← DR startup/reset/interval/step
│       ├── CommandManager ← UniformVelocityCommand (vx,vy,wz)
│       ├── ActionManager ← JointPositionAction → PD actuator
│       ├── ObservationManager ← actor(noise) + critic(privileged)
│       ├── RewardManager ← 10+ weighted terms
│       ├── TerminationManager ← timeout, fallen, out-of-bounds
│       └── CurriculumManager ← terrain levels + velocity stages
│
├── RslRlVecEnvWrapper(env)
│   └── obs_dict → TensorDict, dones = terminated | truncated
│
├── MjlabOnPolicyRunner(env, agent_cfg)  ← extends rsl_rl.OnPolicyRunner
│   └── PPO algorithm (rsl_rl)
│       ├── Actor MLP(128,128,128) + GaussianDistribution
│       └── Critic MLP(128,128,128)
│
└── runner.learn(max_iterations=300)
    ├── COLLECT: 24 steps × 4096 envs
    ├── GAE: γ=0.99, λ=0.95
    ├── UPDATE: 5 epochs × 4 mini-batches
    └── SAVE: .pt + .onnx + W&B upload
```
