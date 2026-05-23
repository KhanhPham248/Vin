# Kế Hoạch Phân Tích Chi Tiết Codebase `mjlab`

## Tổng Quan

**mjlab** là một framework RL (Reinforcement Learning) cho robot, kết hợp API của [Isaac Lab](https://github.com/isaac-sim/IsaacLab) với [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) — phiên bản GPU-accelerated của MuJoCo. Framework chạy **hàng ngàn môi trường song song trên GPU** để train policy cho robot thực tế (humanoid, quadruped).

> [!IMPORTANT]
> **Thuật toán training chính:** PPO (Proximal Policy Optimization) — được implement qua thư viện `rsl-rl-lib==5.2.0` (RSL-RL).
> Network architecture: Actor-Critic MLP, hỗ trợ RNN và CNN encoder.

---

## Kiến Trúc Tổng Thể

```
mjlab/
├── sim/              # [TẦng 1 - Thấp nhất] MuJoCo + Warp GPU physics engine
├── scene/            # [Tầng 2] Quản lý thế giới vật lý: terrain, robot entities
├── entity/           # [Tầng 3] Robot / object model (joints, bodies, sensors)
├── actuator/         # [Tầng 3] Motor model: PD, DC, Learned actuator
├── sensor/           # [Tầng 3] Raycast, contact, camera, IMU sensors
├── terrains/         # [Tầng 3] Sinh địa hình (flat, rough, stairs...)
├── managers/         # [Tầng 4] Quản lý MDP: obs, action, reward, event...
├── envs/             # [Tầng 5] ManagerBasedRlEnv — vòng lặp chính của RL
├── rl/               # [Tầng 6] Wrapper, Runner (PPO training loop)
├── tasks/            # [Tầng 7 - Cao nhất] Task-level config: velocity, tracking
├── scripts/          # Entry points: train.py, play.py, demo.py
└── asset_zoo/        # Robot MJCF models: Unitree G1, Go1, i2rt
```

---

## Giai Đoạn 1 — Tổng Quan Kiến Trúc & Triết Lý Thiết Kế

### Mục tiêu
Hiểu "why" và "how" ở cấp độ hệ thống.

### Việc cần làm

#### 1.1 Đọc tài liệu gốc
- [ ] `README.md` — mô tả mục đích, cách dùng, ví dụ training
- [ ] `CITATION.cff` — xác nhận đây là paper research (arxiv 2601.22074)
- [ ] `CONTRIBUTING.md` + `CLAUDE.md` — hiểu quy tắc dev workflow
- [ ] `pyproject.toml` — **danh sách dependencies quan trọng**:
  - `mujoco-warp` (từ Google DeepMind, dùng GPU CUDA/Warp)
  - `rsl-rl-lib==5.2.0` — **đây là thư viện PPO chính**
  - `warp-lang` — NVIDIA Warp, GPU kernel programming
  - `torch>=2.7.0` — PyTorch
  - `tyro` — CLI config system
  - `wandb` — experiment logging
  - `torchrunx` — multi-GPU training launcher
  - `onnxscript` — export policy sang ONNX cho deployment

#### 1.2 Hiểu luồng training cơ bản

```
train.py
  ├── load task config (env_cfg + agent_cfg)
  ├── create ManagerBasedRlEnv(env_cfg)
  ├── wrap với RslRlVecEnvWrapper
  ├── create MjlabOnPolicyRunner (extends rsl_rl.OnPolicyRunner)
  └── runner.learn(num_iterations)  ← PPO loop bên trong rsl_rl
```

**Files cần đọc:**
- `src/mjlab/scripts/train.py` — entry point, xem cách nó parse args, chọn task, launch GPU
- `src/mjlab/scripts/play.py` — inference/evaluation loop

---

## Giai Đoạn 2 — Phân Tích Module `sim/` và `scene/`

### 2.1 Module `sim/` — Physics Engine Layer

**Files:**
- `sim/sim.py` (~19KB) — **core simulation wrapper**
- `sim/sim_data.py` (~7.6KB) — data structures cho state
- `sim/randomization.py` — domain randomization APIs

**Câu hỏi cần trả lời:**
- `Simulation` class wrap MuJoCo Warp như thế nào?
- `sim.step()` vs `sim.forward()` vs `sim.sense()` khác nhau gì?
- Làm thế nào để chạy N environments song song trên GPU?
- `expand_model_fields()` làm gì trong domain randomization?
- `NaN guard` hoạt động như thế nào?

**Concepts cần nắm:**
- `mj_model` (CPU) vs `model` (GPU Warp) — hai dạng model tồn tại song song
- `data.nworld` = `num_envs` — số lượng parallel worlds
- Physics timestep: `0.005s` (200Hz), env step = `decimation * physics_dt`

### 2.2 Module `scene/` — World Management

**Files:**
- `scene/scene.py` (~9KB)
- `scene/scene.xml` — base XML spec

**Câu hỏi cần trả lời:**
- `Scene` class quản lý `entities` (robots, objects) và `sensors` như thế nào?
- `scene.reset(env_ids)` làm gì?
- `scene.write_data_to_sim()` — tại sao cần bước này?
- `SceneCfg` có những field gì quan trọng?

---

## Giai Đoạn 3 — Phân Tích Module `entity/`, `actuator/`, `sensor/`

### 3.1 Module `entity/` — Robot Model

**Câu hỏi cần trả lời:**
- `Entity` class đại diện cho một robot/object như thế nào?
- `EntityData` chứa những gì (joint_pos, joint_vel, body_pos...)?
- `entity.indexing` — cơ chế index joint/body/ctrl toàn cục vs cục bộ?

### 3.2 Module `actuator/` — Motor Models

**Files:**
| File | Loại actuator |
|------|---------------|
| `actuator.py` | Base class — `ActuatorCfg`, `Actuator`, `ActuatorCmd` |
| `pd_actuator.py` | PD controller: `τ = kp*(q_des - q) + kd*(qd_des - qd)` |
| `dc_actuator.py` | DC motor model với back-EMF |
| `learned_actuator.py` | Neural network-based actuator model |
| `builtin_actuator.py` | Dùng built-in MuJoCo actuator |
| `xml_actuator.py` | Actuator từ XML spec |
| `builtin_group.py` | Fused group cho hiệu năng |

**Câu hỏi cần trả lời:**
- PD actuator tính torque như thế nào? (công thức cụ thể)
- `delay_min_lag` / `delay_max_lag` — tại sao quan trọng cho sim-to-real?
- `learned_actuator` — dùng neural net để model motor dynamics?
- Tại sao có cả `command_field` (position/velocity/effort)?

### 3.3 Module `sensor/` — Sensory Systems

**Files:**
| File | Loại sensor |
|------|-------------|
| `builtin_sensor.py` | IMU, joint encoders, root state |
| `contact_sensor.py` | Foot contact, contact forces |
| `raycast_sensor.py` | Lidar/height scan raycast |
| `camera_sensor.py` | RGB camera |
| `terrain_height_sensor.py` | Local terrain height map |
| `sensor_context.py` | Fused GPU kernel (sense_graph) |

**Câu hỏi cần trả lời:**
- `SensorContext` và `sense_graph` dùng Warp kernel như thế nào?
- Contact sensor tracking `air_time` (thời gian chân trên không) để tính reward?
- Raycast sensor dùng để scan terrain height?

### 3.4 Module `terrains/` — Terrain Generation

**Files:**
- `terrain_generator.py` — procedural terrain generation
- `primitive_terrains.py` (~53KB) — flat, slope, stairs, discrete obstacles
- `heightfield_terrains.py` (~31KB) — perlin noise terrain

**Câu hỏi cần trả lời:**
- Curriculum learning terrain hoạt động thế nào?
- `max_init_terrain_level` — robot bắt đầu ở terrain dễ, dần dần khó hơn?
- Làm thế nào terrain được spawn cho từng environment?

---

## Giai Đoạn 4 — Phân Tích Module `managers/` — Trái Tim của MDP

Đây là layer quan trọng nhất, implement pattern **Manager-Based MDP** từ Isaac Lab.

### Manager nào làm gì?

| Manager | File | Vai trò |
|---------|------|---------|
| `EventManager` | `event_manager.py` | Domain randomization, resets |
| `CommandManager` | `command_manager.py` | Tạo velocity commands cho robot |
| `ActionManager` | `action_manager.py` | Xử lý action từ policy → actuator |
| `ObservationManager` | `observation_manager.py` | Thu thập obs → vector cho policy |
| `RewardManager` | `reward_manager.py` | Tính tổng reward từ nhiều terms |
| `TerminationManager` | `termination_manager.py` | Kiểm tra điều kiện kết thúc episode |
| `CurriculumManager` | `curriculum_manager.py` | Điều chỉnh độ khó theo tiến độ |
| `MetricsManager` | `metrics_manager.py` | Logging các metric phụ |
| `RecorderManager` | `recorder_manager.py` | Ghi rollout data |

**Câu hỏi cần trả lời cho từng manager:**

#### ObservationManager (`observation_manager.py` — 19KB)
- Làm thế nào nó concatenate nhiều `ObservationTermCfg` thành một vector?
- `ObservationGroupCfg` với `enable_corruption=True` — thêm noise như thế nào?
- `observation_history` — stack nhiều timestep obs (cho RNN)?
- `observation_delay` — model sensor latency?
- Tại sao có hai group `actor` vs `critic`? (actor có noise, critic không)

#### EventManager (`event_manager.py` — 15KB)
- 3 modes: `startup`, `reset`, `interval` — mỗi mode khi nào trigger?
- `domain_randomization_fields` — những gì được random?
- Hàm DR: `geom_friction`, `body_com_offset`, `encoder_bias`

#### RewardManager (`reward_manager.py`)
- `scale_by_dt` — tại sao nhân với timestep?
- Reward terms trong velocity task:
  - `track_linear_velocity` (weight=2.0)
  - `track_angular_velocity` (weight=2.0)
  - `upright` (weight=1.0)
  - `pose` (weight=1.0) — phạt deviation khỏi default pose
  - `dof_pos_limits` (weight=-1.0) — phạt quá giới hạn
  - `action_rate_l2` (weight=-0.1) — phạt thay đổi action đột ngột
  - `air_time` — thưởng khi chân có lift-off đúng cách
  - `foot_clearance` (weight=-2.0) — phạt khi chân không lift đủ cao
  - `foot_slip` (weight=-0.1) — phạt khi chân bị trượt

---

## Giai Đoạn 5 — Phân Tích `envs/` và Vòng Lặp RL

### 5.1 `ManagerBasedRlEnv` — Core Environment Loop

**File:** `envs/manager_based_rl_env.py` (~591 lines)

**Vòng lặp `step()` — cần hiểu rõ thứ tự:**

```python
def step(action):
    # 1. Process action (scale, clip)
    action_manager.process_action(action)

    # 2. Decimation loop (4 physics steps per env step)
    for _ in range(decimation):  # 4x
        action_manager.apply_action()    # → actuator → ctrl
        scene.write_data_to_sim()
        sim.step()                       # MuJoCo Warp GPU step
        scene.update(dt)
        metrics_manager.compute_substep()

    # 3. Check terminations & compute rewards
    termination_manager.compute()
    reward_manager.compute(dt=step_dt)

    # 4. Auto-reset terminated envs
    reset_env_ids = reset_buf.nonzero()
    _reset_idx(reset_env_ids)

    # 5. Single forward() call — update derived quantities
    sim.forward()                        # ← key design decision

    # 6. Step events, sense, compute observations
    event_manager.apply(mode="step")
    sim.sense()
    observation_manager.compute()

    return obs, reward, terminated, truncated, info
```

**Câu hỏi cần trả lời:**
- Tại sao `sim.forward()` chỉ được gọi một lần sau khi reset?
- `decimation=4` có nghĩa gì về control frequency? (200Hz physics → 50Hz policy)
- `is_finite_horizon` ảnh hưởng đến bootstrap value thế nào?
- `auto_reset=True` vs `False` — khi nào dùng mỗi loại?

### 5.2 Module `envs/mdp/` — Shared MDP Functions

**Files:**
- `observations.py` — joint_pos_rel, joint_vel_rel, projected_gravity, height_scan
- `rewards.py` — action_rate_l2, joint_pos_limits
- `events.py` (~21KB) — reset functions, push_robot, DR functions
- `actions/` — JointPositionAction, DifferentialIkAction
- `curriculums.py` — terrain levels, command velocity scheduling
- `terminations.py` — time_out, bad_orientation, out_of_bounds
- `dr/` — domain randomization functions

---

## Giai Đoạn 6 — Phân Tích Thuật Toán Training (PPO + RSL-RL)

### 6.1 PPO Configuration

**File:** `rl/config.py`

```python
# Actor network
hidden_dims: (128, 128, 128)  # 3 lớp hidden
activation: "elu"
obs_normalization: False

# PPO hyperparameters
num_learning_epochs: 5       # Số lần update trên mỗi batch
num_mini_batches: 4          # mini_batch_size = num_envs * steps / 4
learning_rate: 1e-3
gamma: 0.99                  # Discount factor
lam: 0.95                    # GAE lambda
entropy_coef: 0.005          # Entropy regularization
desired_kl: 0.01             # Adaptive LR target
clip_param: 0.2              # PPO clip epsilon
max_grad_norm: 1.0
schedule: "adaptive"         # Adaptive LR dựa trên KL divergence
```

### 6.2 Luồng PPO Training (qua RSL-RL)

```
MjlabOnPolicyRunner.learn()
    ├── COLLECTION PHASE (num_steps_per_env=24 steps)
    │   └── for step in 24:
    │       ├── obs → actor_network → action (stochastic)
    │       ├── env.step(action) → next_obs, reward, done
    │       └── store (obs, action, reward, done, value, log_prob)
    │
    ├── ADVANTAGE ESTIMATION (GAE)
    │   └── A_t = Σ (γλ)^k * δ_{t+k}  where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    │
    └── UPDATE PHASE (5 epochs × 4 mini-batches)
        ├── Policy loss: L_clip = E[min(r_t * A_t, clip(r_t, 1±ε) * A_t)]
        ├── Value loss: L_value = (V_pred - V_target)^2
        ├── Entropy bonus: L_entropy = H(π)
        └── Total: L = -L_clip + c1*L_value - c2*L_entropy
```

### 6.3 RSL-RL Interface

**File:** `rl/runner.py` — `MjlabOnPolicyRunner`

Extends `rsl_rl.runners.OnPolicyRunner` với:
- `save()` — lưu checkpoint + auto-export ONNX + upload W&B
- `load()` — hỗ trợ migrate legacy checkpoint format
- ONNX export với `dynamo=False` (tránh warning torch>=2.9)

**File:** `rl/vecenv_wrapper.py` — `RslRlVecEnvWrapper`

Wrap `ManagerBasedRlEnv` để tương thích với RSL-RL interface:
- Map `obs_buf["actor"]` → actor obs
- Map `obs_buf["critic"]` → critic obs
- Xử lý `obs_groups` config

### 6.4 Curriculum Learning

```python
curriculum = {
    "terrain_levels": CurriculumTermCfg(func=mdp.terrain_levels_vel),
    "command_vel": CurriculumTermCfg(
        func=mdp.commands_vel,
        velocity_stages=[
            {"step": 0,           "lin_vel_x": (-1.0, 1.0)},
            {"step": 5000 * 24,   "lin_vel_x": (-1.5, 2.0)},  # Tăng tốc độ sau 5000 iters
            {"step": 10000 * 24,  "lin_vel_x": (-2.0, 3.0)},  # Tăng tiếp sau 10000 iters
        ],
    ),
}
```

---

## Giai Đoạn 7 — Phân Tích Tasks Cụ Thể

### 7.1 Task 1: Velocity Tracking (task chính)

**Files:**
- `tasks/velocity/velocity_env_cfg.py` — base config
- `tasks/velocity/config/g1/` — G1-specific config
- `tasks/velocity/config/go1/` — Go1-specific config
- `tasks/velocity/mdp/` — velocity-specific reward/obs functions
- `tasks/velocity/rl/runner.py` — `VelocityOnPolicyRunner`

**Mục tiêu:** Train Unitree G1 humanoid chạy theo velocity command (vx, vy, ωz)

**Observation vector (actor):**
| Term | Dim | Noise |
|------|-----|-------|
| base_lin_vel (IMU) | 3 | ±0.5 |
| base_ang_vel (IMU) | 3 | ±0.2 |
| projected_gravity | 3 | ±0.05 |
| joint_pos (relative to default) | N_joints | ±0.01 |
| joint_vel | N_joints | ±1.5 |
| last_action | N_joints | - |
| command (vx, vy, ωz) | 3 | - |
| height_scan (terrain map) | ~187 | ±0.1 |

**Action:** joint position targets (delta từ default pose, scale=0.5)

### 7.2 Task 2: Motion Tracking (Imitation Learning)

**Files:**
- `tasks/tracking/tracking_env_cfg.py`
- `tasks/tracking/mdp/` — motion command, tracking rewards
- `scripts/csv_to_npz.py` — convert motion data

**Mục tiêu:** Imitate reference motions (captured from real robot hoặc motion retargeting)

### 7.3 Task 3: Cartpole (Sanity Check)

**Files:** `tasks/cartpole/`

**Mục tiêu:** Classic CartPole — dùng để verify framework, không phải robot task.

---

## Giai Đoạn 8 — Domain Randomization (Sim-to-Real Gap)

**Tại sao quan trọng:** Để policy train trong sim có thể deploy lên robot thực tế.

**DR được apply ở `startup` (một lần khi reset):**
| DR Term | Ý nghĩa |
|---------|---------|
| `geom_friction` | Random foot friction [0.3, 1.2] |
| `encoder_bias` | Random joint encoder offset ±0.015 rad |
| `body_com_offset` | Random center-of-mass shift ±2.5cm |
| `push_robot` (interval) | Push robot với random velocity mỗi 1-3 giây |

**Files cần đọc:**
- `envs/mdp/dr/` — các DR functions
- `test_domain_randomization.py` (~60KB) — test toàn diện, rất hữu ích để hiểu

---

## Giai Đoạn 9 — Tooling & Deployment

### 9.1 Scripts Entry Points

| Script | Lệnh | Mục đích |
|--------|------|---------|
| `train.py` | `uv run train TaskId` | Train PPO |
| `play.py` | `uv run play TaskId` | Evaluate/visualize |
| `demo.py` | `uv run demo` | Quick demo |
| `list_envs.py` | `uv run list-envs` | List all registered tasks |
| `nan_viz.py` | `uv run viz-nan` | Debug NaN issues |
| `export_scene.py` | `uv run export-scene` | Export scene |

### 9.2 ONNX Export

- Policy được export sang ONNX sau mỗi save checkpoint
- Metadata được attach vào ONNX file (qua `exporter_utils.py`)
- Dùng để deploy lên robot thực tế (không cần PyTorch)

### 9.3 Multi-GPU Training

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

Dùng `torchrunx` để launch distributed training. Mỗi GPU chạy một process với `LOCAL_RANK` khác nhau.

---

## Thứ Tự Đọc Code Được Đề Xuất

### Tuần 1 — Foundation
1. `README.md` → `pyproject.toml` → `CLAUDE.md`
2. `scripts/train.py` (hiểu entry point)
3. `envs/manager_based_rl_env.py` (vòng lặp chính)
4. `rl/config.py` + `rl/runner.py` (PPO config)

### Tuần 2 — Core Modules
5. `sim/sim.py` (physics engine)
6. `scene/scene.py`
7. `actuator/actuator.py` → `pd_actuator.py`
8. `sensor/builtin_sensor.py` → `contact_sensor.py` → `raycast_sensor.py`

### Tuần 3 — Managers & MDP
9. `managers/observation_manager.py` (dài nhất, 19KB)
10. `managers/event_manager.py` (domain randomization)
11. `managers/reward_manager.py`
12. `envs/mdp/events.py`

### Tuần 4 — Tasks & Integration
13. `tasks/velocity/velocity_env_cfg.py` (xem cách assemble task)
14. `tasks/velocity/config/g1/` (G1-specific customization)
15. `tasks/tracking/tracking_env_cfg.py`
16. `tests/test_variants.py` (~55KB) — test nhiều robot config, rất informative

---

## Câu Hỏi Interview Cần Chuẩn Bị

1. **Tại sao dùng PPO thay vì SAC hay TD3?** → On-policy, stable với parallel envs, RSL-RL đã chứng minh với legged robots
2. **Decimation có ý nghĩa gì?** → Policy chạy 50Hz, physics 200Hz — tách biệt control frequency với physics stability
3. **Tại sao Actor và Critic có obs khác nhau?** → Critic có thêm privileged info (contact forces, true foot height) không nhiễu để học value function tốt hơn
4. **Domain Randomization giúp gì?** → Thu hẹp sim-to-real gap, tạo ra policy robust
5. **Curriculum Learning hoạt động thế nào?** → Terrain levels tăng khi robot thành công; command velocity tăng dần theo số bước train
6. **Tại sao cần `sim.forward()` sau reset?** → MuJoCo cần recompute derived quantities (xpos, xquat) sau khi state bị modified trực tiếp
7. **ONNX export để làm gì?** → Deploy policy lên robot thực tế mà không cần PyTorch

---

> [!NOTE]
> Đây là framework production-grade được dùng trong nghiên cứu thực sự. Tham khảo thêm paper gốc: [arxiv 2601.22074](https://arxiv.org/abs/2601.22074) để hiểu context nghiên cứu.
