# Nhật Ký Sửa Đổi #001: GPU Solver & Tối Ưu Hóa Hàm Thưởng HU_D03

**Ngày sửa đổi:** 24/05/2026
**Mục tiêu:** 
1. Khắc phục lỗi biên dịch LTO của Nvidia Warp trên GPU Quadro P2000 (kiến trúc Pascal sm_61).
2. Giải quyết xung đột phần thưởng tư thế giữa khớp chủ động (actuated) và khớp thụ động (passive/rods).
3. Loại bỏ hiện tượng rò rỉ phần thưởng vận tốc (velocity reward leaking) khiến robot bị kẹt ở nghiệm cục bộ đứng im/đi chậm.

---

## 1. Tự Động Nhận Diện GPU & Chọn Solver Phù Hợp
- **Vấn đề:** Môi trường mặc định sử dụng solver `Newton` của `mujoco_warp`. Solver này sử dụng kernel tính ma trận song song nâng cao (`wp.tile_matmul`) vốn yêu cầu GPU kiến trúc Volta/Turing trở lên (sm_70+). Trình biên dịch LTO bị lỗi crash trên GPU cũ Pascal sm_61 (Quadro P2000) của máy cá nhân.
- **Giải pháp:** Viết hàm tự động quét Compute Capability của GPU qua PyTorch. Nếu phát hiện card cũ (< sm_70), hệ thống tự động đổi solver MuJoCo từ `"newton"` sang `"cg"` (Conjugate Gradient) để chạy mượt mà trên máy cá nhân, trong khi vẫn giữ `"newton"` hiệu năng cao trên Cloud (GPU L4/A100).
- **Code thay đổi:**
  ```python
  def _get_optimal_solver() -> str:
      import os
      if "MJLAB_SOLVER" in os.environ:
          return os.environ["MJLAB_SOLVER"]
      try:
          import torch
          if torch.cuda.is_available():
              major, _ = torch.cuda.get_device_capability(0)
              if major < 7:  # Volta (sm_70) là tối thiểu cho Newton/Warp LTO
                  return "cg"
      except Exception:
          pass
      return "newton"
  ```
  *(Áp dụng cho cả `hu_d03_flat_env_cfg` và `hu_d03_rough_env_cfg`)*

---

## 2. Loại Bỏ Khớp Thụ Động Khỏi Phạt Tư Thế (Pose Reward Masking)
- **Vấn đề:** 
  1. G1 chỉ có 29 khớp chủ động nên chọn mặc định `".*"` cho phần thưởng tư thế `"pose"` hoạt động tốt. HU_D03 có tới 55 khớp (31 chủ động, 24 thụ động). Hàm phạt cũ đè phạt lên cả 24 khớp thụ động này. Do các thanh liên kết 4-bar và cổ chân bắt buộc phải xoay khi chân di chuyển, robot chịu điểm phạt tư thế liên tục rất lớn nếu bước đi $\rightarrow$ Robot chọn đứng im làm nghiệm tối ưu.
  2. G1 có các khớp eo trực tiếp (`waist_roll`, `waist_pitch`). Nhưng HU_D03 sử dụng liên kết song song được kéo bởi các khớp chủ động `waist_yaw`, `waist_A`, `waist_B`. Khi siết chặt khớp về 31 khớp chủ động, việc giữ nguyên các mẫu `r"waist_roll.*"` và `r"waist_pitch.*"` trong cấu hình sai số `std_walking` và `std_running` gây lỗi crash `ValueError: Not all regular expressions are matched` do các khớp eo xoay/gập trực tiếp này không tồn tại trong danh sách khớp chủ động của HU_D03.
- **Giải pháp:** 
  1. Tạo hằng số lọc khớp chủ động `ACTUATED_JOINT_NAMES` và gán vào `joint_names` của phần thưởng `"pose"`.
  2. Loại bỏ khóa `r".*ankle.*"` (vì cổ chân là khớp thụ động) và khóa `r"waist_roll.*"`, `r"waist_pitch.*"` khỏi từ điển sai số `std_walking` và `std_running` trong cả 2 cấu hình Flat và Rough.
- **Danh sách khớp được bảo vệ (Chỉ phạt tư thế trên các khớp này):**
  ```python
  ACTUATED_JOINT_NAMES = (
      r".*_hip_pitch_joint",
      r".*_hip_roll_joint",
      r".*_hip_yaw_joint",
      r".*_knee_joint",
      r".*_A_achilles_joint",
      r".*_B_achilles_joint",
      r"waist_yaw_joint",
      r"waist_A_joint",
      r"waist_B_joint",
      r".*_shoulder_pitch_joint",
      r".*_shoulder_roll_joint",
      r".*_shoulder_yaw_joint",
      r".*_elbow_joint",
      r".*_wrist_yaw_joint",
      r".*_wrist_pitch_joint",
      r".*_hand_yaw_joint",
      r"head_yaw_joint",
      r"head_pitch_joint",
  )
  ```

---

## 3. Siết Chặt Sai Số Bám Vận Tốc (Eliminate Reward Leaking)
- **Vấn đề:** Tham số độ lệch chuẩn `std = 0.5` cho tracking vận tốc quá lỏng lẻo. Robot đứng im hoàn toàn vẫn ăn gian được tới **36.8%** lượng điểm thưởng bám vận tốc mà không cần chịu rủi ro di chuyển làm ngã robot.
- **Giải pháp:** Siết chặt đáng kể độ lệch chuẩn `std` trong hàm Gaussian bám vận tốc ở cả môi trường phẳng và gồ ghề:
  *   Bám vận tốc dài (`lin_vel`): Siết từ `0.5` xuống **`0.15`** ($std^2 \approx 0.0225$).
  *   Bám vận tốc góc (`ang_vel`): Siết từ `0.5` xuống **`0.25`** ($std^2 \approx 0.0625$).
  *(Ở cấu hình mới, nếu có lệnh di chuyển mà robot đứng im, điểm thưởng nhận được gần như bằng 0, buộc PPO phải tìm chính sách bước đi).*

---

## 📂 File Đã Cập Nhật
*   **`src/hu_d03_locomotion/tasks/velocity_env_cfg.py`**
