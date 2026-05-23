# HU_D03 Locomotion

Velocity-tracking locomotion policy cho robot **HU_D03** humanoid, xây dựng trên framework [mjlab](../mjlab/mjlab-main).

## Cấu Trúc

```
HU_D03_locomotion/
├── assets/
│   └── robots/hu_d03/
│       ├── xmls/hu_d03.xml        # MJCF (mjlab-compatible, stripped)
│       └── meshes -> symlink      # STL meshes từ humanoid-description
├── src/hu_d03_locomotion/
│   ├── robots/
│   │   └── hu_d03_constants.py    # PD gains, keyframe, collision, action_scale
│   └── tasks/
│       ├── velocity_env_cfg.py    # flat + rough env configs
│       ├── rl_cfg.py              # PPO runner configs
│       └── __init__.py            # task registration
├── scripts/
│   ├── train.py                   # training entry point
│   └── play.py                    # evaluation entry point
├── logs/                          # training outputs (gitignored)
├── configs/                       # override YAML configs (optional)
└── pyproject.toml
```

## Đặc Điểm HU_D03 Khác G1

| | G1 | HU_D03 |
|--|----|----|
| Root body | `pelvis` | `base_link` |
| Ankle control | `ankle_pitch/roll` direct | **Achilles 4-bar** (`A/B_achilles_joint`) |
| Waist | direct joints | **Waist 4-bar** (`waist_A/B_joint`) |
| Foot sites | pre-existing | **Thêm vào XML** (`<site name="left_foot">`) |

## Setup

```bash
cd HU_D03_locomotion
conda activate Vin
uv sync
```

## Chạy

```bash
# Smoke test — zero agent đứng yên, kiểm tra env init
uv run python scripts/play.py Mjlab-Velocity-Flat-HuD03 --agent zero

# Random agent — kiểm tra physics không explode
uv run python scripts/play.py Mjlab-Velocity-Flat-HuD03 --agent random

# Training — flat terrain trước
uv run python scripts/train.py Mjlab-Velocity-Flat-HuD03 \
    --env.scene.num-envs 1024

# Training — rough terrain (sau khi flat converge)
uv run python scripts/train.py Mjlab-Velocity-Rough-HuD03 \
    --env.scene.num-envs 2048 \
    --agent.load-run <flat_run_name>
```

## Calibration Cần Thiết

### 1. Keyframe (Standing Pose)
Nếu robot không đứng được khi init:
```bash
# Mở MuJoCo viewer với XML gốc
cd ../humanoid-description/humanoid-description-master/HU_D03_description
python view_mujoco.py
```
Điều chỉnh joint values đến khi robot đứng tự nhiên → copy vào `HOME_KEYFRAME` trong `hu_d03_constants.py`.

### 2. Achilles Joint Mapping
Joints `left_A_achilles_joint` / `left_B_achilles_joint` điều khiển ankle qua 4-bar linkage:
- A_achilles ≈ controls ankle pitch component
- B_achilles ≈ controls ankle roll component
- Cả hai ở `0.0` = ankle flat

### 3. PD Gains
Verify bằng zero-action policy:
```bash
uv run python scripts/play.py Mjlab-Velocity-Flat-HuD03 --agent zero
```
Robot phải đứng ~2–3 giây. Nếu sụp ngay → tăng `kp`; nếu rung → giảm `kp`, tăng `kd`.

## TensorBoard

```bash
tensorboard --logdir logs/
```
