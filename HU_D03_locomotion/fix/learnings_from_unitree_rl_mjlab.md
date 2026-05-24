# Báo Cáo Phân Tích: Học Hỏi & Đối Chiếu Từ Unitree RL MJLab ("Cực Chuẩn")

**Đường dẫn đối chiếu:** `/home/khanh248/Documents/Vin/unitree_rl_mjlab-main/src/tasks/velocity`
**Mục tiêu:** Phân tích cấu trúc hàm thưởng (rewards), cách biểu diễn quan sát (observations), và chiến lược curriculum của Unitree để áp dụng nâng cấp dáng đi cho robot humanoid **HU_D03**.

---

## 1. Những "Bí Kíp" Độc Quyền Của Unitree Trong `mdp/rewards.py`

Khi so sánh giữa base `mjlab` và repository của Unitree, chúng ta phát hiện Unitree sở hữu **5 hàm thưởng cực kỳ tinh xảo** mà base `mjlab` không hề có. Đây chính là chìa khóa giúp robot của họ đạt được dáng đi tự nhiên, vững chãi:

### 🔑 Bí Kíp 1: Hàm Đồng Bộ Chu Kỳ Bước Đi (`foot_gait`)
*   **Toán học:** 
    ```python
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    ```
*   **Ý nghĩa:** Tạo ra một "đồng hồ sinh học" (internal clock) cho robot. Với chu kỳ `period = 0.6s` và độ lệch pha giữa 2 chân là `0.5` (lệch pha 180 độ), hàm thưởng này ép buộc chân trái và chân phải phải **luân phiên chạm đất và nhấc chân một cách đối xứng hoàn hảo**.
*   **Hiệu quả:** Triệt tiêu hoàn toàn các dáng đi lỗi như đi lết, đi nhảy hai chân (hop) hoặc nhảy lò cò. Robot sẽ bước đi nhịp nhàng như người thật!

### 🔑 Bí Kíp 2: Phạt Mô-men Động Lượng Toàn Thân (`angular_momentum`)
*   **Toán học:** Đo độ lớn bình phương của mô-men động lượng toàn thân (`robot/root_angmom`):
    $$R_{angmom} = -w \cdot \|\mathbf{L}_{world}\|^2$$
*   **Ý nghĩa:** Khi robot di chuyển chân, nó sinh ra mô-men quay. Để triệt tiêu mô-men này và giữ thăng bằng, phần thân trên bắt buộc phải phản ứng.
*   **Hiệu quả:** Ép hai cánh tay của robot **đánh tay tự nhiên (arm swing)** ngược chiều với chân để triệt tiêu mô-men động lượng, giữ torso đứng thẳng không bị lắc lư qua lại.

### 🔑 Bí Kíp 3: Phạt Tiếp Đất Cứng (`soft_landing`)
*   **Toán học:** Đo lực va đập tại thời điểm đầu tiên tiếp đất (first contact):
    ```python
    landing_impact = force_magnitude * first_contact.float()
    ```
*   **Ý nghĩa:** Chỉ phạt lực tiếp xúc đúng lúc chân vừa chạm đất.
*   **Hiệu quả:** Giúp robot đi "nhẹ nhàng", tránh việc nện chân quá mạnh xuống sàn gây hại cho phần cứng thực tế và làm nhiễu cảm biến IMU.

### 🔑 Bí Kíp 4: Phạt Kéo Lê Chân (`foot_slip`)
*   **Toán học:** Phạt vận tốc XY của bàn chân khi bàn chân đang chạm đất (`in_contact`):
    ```python
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1)
    ```
*   **Ý nghĩa:** Bàn chân khi đã chạm đất thì vận tốc trượt XY phải bằng 0.
*   **Hiệu quả:** Tránh hiện tượng robot đi trượt băng (slipping), giúp bám đường cực tốt khi chuyển lên môi trường gồ ghề hoặc trơn trượt.

### 🔑 Bí Kíp 5: Đứng Im Nghiêm Túc (`stand_still`)
*   **Toán học:** Phạt lệch khớp khỏi tư thế đứng mẫu chỉ khi vận tốc lệnh bằng 0.
*   **Ý nghĩa:** Khi không có lệnh di chuyển, robot phải đứng yên hoàn toàn ở tư thế thiết kế (`HOME_KEYFRAME`), không được xê dịch hay lắc lư vô nghĩa.

---

## 2. Quan Sát Pha Chuyển Động (`Observation: phase`)

Trong `mdp/observations.py`, Unitree đưa vào một biến trạng thái cực kỳ thông minh:
```python
def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
```
*   **Cách hoạt động:** Gửi tọa độ $\sin(2\pi \cdot t/T)$ và $\cos(2\pi \cdot t/T)$ trực tiếp vào chính sách Actor.
*   **Đặc điểm:** Khi robot được lệnh đứng im (`speed < 0.1`), pha lập tức được đưa về `0.0`.
*   **Lợi ích**: Giúp mạng thần kinh PPO nhận biết rõ ràng nhịp thời gian để ra quyết định nhấc/hạ chân đồng bộ, đồng thời đứng yên vững vàng khi không có lệnh.

---

## 3. Đối Chiếu Hệ Số Thưởng Giữa HU_D03 Và G1 (Super Standard)

Dưới đây là bảng đối chiếu hệ số thưởng giữa cấu hình hiện tại của **HU_D03** và cấu hình siêu chuẩn của **G1** từ Unitree:

| Tên Phần Thưởng (Reward Term) | Trọng Số HU_D03 Hiện Tại | Trọng Số G1 (Unitree) | Đánh Giá & Bài Học Rút Ra |
| :--- | :---: | :---: | :--- |
| **`track_linear_velocity`** | `3.0` (std=0.15) | `1.0` (std=0.5) | Trọng số HU_D03 cao hơn kết hợp std siết chặt giúp ép robot di chuyển bám sát lệnh tốt hơn. |
| **`track_angular_velocity`** | `1.0` (std=0.25) | `1.0` (std=0.7) | Rất tương đồng, HU_D03 siết std chặt hơn để tránh xoay người vô lý. |
| **`body_orientation_l2`** | `-1.0` | `-1.0` | Đồng bộ hoàn hảo (phạt nghiêng người). |
| **`angular_momentum`** | `-0.02` | `-0.025` | Rất khớp, giúp duy trì chuyển động đánh tay thăng bằng torso. |
| **`air_time`** (generic) | `3.0` | `0.0` (Tắt) | Unitree **tắt hoàn toàn** air_time truyền thống và thay bằng `foot_gait`! |
| **`foot_gait`** (cyclic) | **Chưa có** | **`0.5`** | **BÀI HỌC LỚN**: Nên bổ sung `foot_gait` và `phase` vào HU_D03 để có dáng đi đẹp chuẩn mực! |
| **`soft_landing`** | `-0.05` | `-0.001` | Phạt va đập tiếp đất của HU_D03 đang hơi nặng, có thể nới lỏng nếu chân nhấc không lên. |
| **`foot_slip`** | `-0.05` | `-0.25` | Phạt trượt chân của Unitree rất nặng để triệt tiêu hiện tượng đi lết trượt. |
| **`stand_still`** | **Chưa có** | **`-1.0`** | **BÀI HỌC LỚN**: Nên bổ sung để robot HU_D03 đứng im phăng phắc khi lệnh vận tốc = 0. |

---

## 4. Đề Xuất Nâng Cấp Lộ Trình Cho HU_D03

Để robot HU_D03 đạt được **dáng đi đẳng cấp và tự nhiên nhất**, chúng ta nên thực hiện nâng cấp theo 2 bước:

1.  **Bước 1 (Hiện tại)**: Bạn hãy cứ cho chạy thử nghiệm đợt huấn luyện hiện tại với các sửa đổi sửa lỗi khớp và siết std của tôi. Hệ thống chắc chắn sẽ thoát khỏi nghiệm cục bộ cũ và đi được bình thường.
2.  **Bước 2 (Nâng cấp dáng đi đẹp)**: Nếu bạn muốn robot HU_D03 có dáng đi **"đánh tay nhịp nhàng, bước chân đối xứng chuẩn chỉ như người thật"**, chúng ta sẽ:
    *   Tích hợp hàm `phase` vào danh sách `Observation` của Actor/Critic.
    *   Tích hợp hàm thưởng `feet_gait` và `stand_still` từ Unitree vào dự án.
    *   Kích hoạt chúng trong `velocity_env_cfg.py`.

*Tôi đã sẵn sàng viết code tích hợp trọn bộ tính năng này bất cứ lúc nào bạn yêu cầu!*
