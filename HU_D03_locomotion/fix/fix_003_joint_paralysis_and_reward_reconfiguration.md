# Nhật Ký Sửa Đổi #003: Giải Phóng Liệt Khớp & Tái Cấu Trúc Hàm Thưởng HU-D03

**Ngày sửa đổi:** 24/05/2026
**Mục tiêu:** 
1. Khắc phục lỗi cơ học tê liệt khớp gối/hông khiến robot không thể sải bước chân.
2. Sửa lỗi "hack reward" điểm bay chân (`air_time`) và triệt tiêu rung lắc tần số cao.
3. Nới rộng dải Gaussian của thưởng bám vận tốc để tăng cường tín hiệu Gradient trong giai đoạn đầu học đi.

---

## 1. Khắc Phục Lỗi Tê Liệt Khớp (Action Scale Fix)
- **Vấn đề:** Công thức tính tỉ lệ hành động tự động kế thừa từ G1 là `0.25 * effort_limit / stiffness`. Tuy nhiên, vì khớp hông/gối của HU-D03 cực kỳ khỏe (Stiffness = `602` Nm/rad), công thức này trả về tỉ lệ hành động chỉ **`0.05` rad (tương đương 2.8 độ)**. Điều này khiến robot bị khóa cứng chân, hoàn toàn không có khả năng cơ học để co chân/sải bước, gây ra sai số vận tốc khổng lồ (`error_vel_xy = 0.58` m/s) và flailing dữ dội để cố thoát thế kẹt.
- **Giải pháp:** Cấu hình thủ công tỉ lệ hành động thực tế đã được kiểm chứng trên các robot humanoid chân lớn:
  *   Các khớp hông, gối, cổ chân Achilles, eo: **`0.25` rad (14.3 độ)**. Khoảng cách này cho phép robot duỗi gối và bước đi thoải mái.
  *   Các khớp tay và đầu: **`0.30` rad**.
- **Code thay đổi trong `hu_d03_constants.py`:**
  ```python
  HU_D03_ACTION_SCALE: dict[str, float] = {}
  for _act in HU_D03_ARTICULATION.actuators:
      assert isinstance(_act, BuiltinPositionActuatorCfg)
      for _n in _act.target_names_expr:
          # Leg joints (hips, knees, ankles) and waist need realistic range (0.25 rad ≈ 14.3 deg)
          if any(keyword in _n for keyword in ("hip", "knee", "achilles", "waist")):
              HU_D03_ACTION_SCALE[_n] = 0.25
          else:
              HU_D03_ACTION_SCALE[_n] = 0.30
  ```

---

## 2. Loại Bỏ Hiện Tượng Hack Reward & Rung Lắc Tần Số Cao
- **Vấn đề:** 
  1. Trọng số thưởng bay chân `air_time = 3.0` quá lớn so với điểm thưởng bám vận tốc (vốn gần bằng 0 do liệt khớp ở mục 1). Robot tìm ra nghiệm cục bộ dễ nhất là ngã ngửa ra sàn hoặc đứng yên giật chân liên tục để ăn gian thưởng bay chân.
  2. Phạt thay đổi hành động đột ngột `action_rate_l2 = -0.01` quá nhẹ khiến chính sách xuất ra hành động rung lắc dữ dội tần số cao.
- **Giải pháp:** 
  1. Giảm trọng số thưởng bay chân `air_time` từ `3.0` xuống **`1.0`** (bằng 1/3 thưởng bám vận tốc `3.0`), buộc robot ưu tiên đi thẳng.
  2. Tăng phạt rung giật `action_rate_l2` lên **`-0.05`** để làm mượt các hành động góc khớp, triệt tiêu flailing chân.
- **Code thay đổi trong `velocity_env_cfg.py`:**
  ```python
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["air_time"].weight = 1.0
  ```

---

## 3. Tinh Chỉnh Dải Gaussian Bám Vận Tốc (Gradient Balance)
- **Vấn đề:** Sai số tiêu chuẩn bám vận tốc đặt quá hẹp (`std = 0.15`) gây triệt tiêu gradient ban đầu, nhưng khi nới lên quá rộng (`0.25`), robot lại chọn đứng im ăn rò rỉ điểm vận tốc.
- **Giải pháp:** Tinh chỉnh về mức tối ưu **`0.20`** cho vận tốc dài và **`0.40`** cho vận tốc góc để vừa giữ gradient tốt vừa giảm rò rỉ điểm khi đứng im xuống sát 0.
- **Code thay đổi trong `velocity_env_cfg.py`:**
  ```python
  cfg.rewards["track_linear_velocity"].params["std"] = 0.20
  cfg.rewards["track_angular_velocity"].params["std"] = 0.40
  ```

---

## 4. Bẻ Gãy Nghiệm Cục Bộ Đứng Im Chèo Chân (Reward Hacking Breakout)
- **Vấn đề:** Robot đứng im tại chỗ (bảo toàn tính mạng, ăn điểm `upright`) và liên tục đạp/chèo một chân lên xuống để "vắt sữa" điểm thưởng `air_time` mà không chịu đi thật.
- **Giải pháp:** 
  1. Giảm mạnh trọng số thưởng bay chân `air_time` từ `1.0` xuống **`0.3`** để triệt tiêu sức hút của việc đạp chân ăn điểm.
  2. Tăng hình phạt trượt chân `foot_slip` từ `-0.05` lên **`-0.20`** để trừng phạt động tác cào chân lê lết dưới mặt đất.
- **Code thay đổi trong `velocity_env_cfg.py`:**
  ```python
  cfg.rewards["air_time"].weight = 0.3
  cfg.rewards["foot_slip"].weight = -0.20
  ```

---

## 📂 File Đã Cập Nhật
*   **`src/hu_d03_locomotion/robots/hu_d03_constants.py`**
*   **`src/hu_d03_locomotion/tasks/velocity_env_cfg.py`**
