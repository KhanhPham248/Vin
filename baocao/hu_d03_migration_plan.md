# Kế Hoạch Chuyển Đổi Thực Tế (Migration Plan): G1 ➜ HU_D03 (Dựa trên lịch sử file README và Git Diff)

Xin lỗi vì trước đó tôi đã đưa ra một bản kế hoạch quá chung chung dựa trên lý thuyết. Dựa trên lịch sử các file gốc, `README.md` và quá trình debug thực tế trong thư mục `HU_D03_locomotion`, đây là **chính xác những gì đã và cần phải làm** để mang robot HU_D03 (hoặc các phiên bản HC_03 tương tự) vào hệ thống `mjlab`:

---

## Giai Đoạn 1: Xử Lý Biến Đổi Khung Xương (Kinematics & XML)

HU_D03 có kiến trúc cơ khí phức tạp hơn Unitree G1 rất nhiều, đặc biệt là ở cơ cấu truyền động gián tiếp (4-bar linkage).

- [ ] **1. Đổi tên Root Body:**
  - Unitree G1 sử dụng `pelvis` làm gốc.
  - Phải đổi gốc của hệ thống nhận diện sang `base_link` cho cấu trúc của HU_D03.
- [ ] **2. Thêm Điểm Neo (Sites) Thủ Công vào XML:**
  - Mô hình CAD của HU_D03 không có sẵn điểm tiếp xúc bàn chân. Phải chủ động code thêm thẻ `<site name="left_foot" ...>` và `<site name="right_foot" ...>` vào `hu_d03.xml` để tính toán Reward (lực đạp đất).
- [ ] **3. Xử Lý Cơ Cấu Mắt Cá Chân (Achilles 4-bar):**
  - G1 dùng `ankle_pitch` và `ankle_roll` trực tiếp.
  - HU_D03 phải map lại thành điều khiển 2 khớp truyền động: `left_A_achilles_joint` và `left_B_achilles_joint`. (Khi cả A và B = `0.0` thì bàn chân phẳng).
- [ ] **4. Xử Lý Cơ Cấu Lưng/Eo (Waist 4-bar):**
  - Thay vì điều khiển trực tiếp lưng như G1, phải map lại thuật toán điều khiển qua 2 khớp `waist_A_joint` và `waist_B_joint`.

---

## Giai Đoạn 2: Xử Lý Lỗi Cảm Biến Hệ Thống (Sensor Debugging)

- [ ] **1. Bổ sung Cảm biến Động lượng góc (`root_angmom`):**
  - Hàm phạt (Penalty) của hệ thống Mjlab yêu cầu tính toán `angular_momentum` để chống vặn mình.
  - Phải chèn thêm thẻ `<subtreeangmom name="root_angmom" body="base_link" />` vào block `<sensor>` trong `hu_d03.xml` (nếu không sẽ bị lỗi `KeyError: 'robot/root_angmom'` khi chạy Train).

---

## Giai Đoạn 3: Calibration Phần Cơ & PID (Robot Constants)

- [ ] **1. Dò tìm Tư thế Đứng (Standing Pose - `HOME_KEYFRAME`):**
  - Phải chạy script `view_mujoco.py` (từ gói `humanoid-description`) để chỉnh tay góc khớp đến khi robot đứng tự nhiên nhất.
  - **Lưu ý Đặc biệt:** Trục xoay khớp của HU_D03 bị ngược ở tay. Khớp khuỷu tay (`elbow_joint`) phải thiết lập giá trị **âm** (`-1.20`) để tay gập về phía trước, thay vì số dương như G1 (sẽ làm tay bẻ gập ngược ra sau lưng).
- [ ] **2. Cân chỉnh PD Gains bằng Zero Agent:**
  - Chạy lệnh `uv run python scripts/play.py Mjlab-Velocity-Flat-HuD03 --agent zero` để mô phỏng robot thả rơi tự do.
  - Cố gắng điều chỉnh `kp` (Độ cứng) và `kd` (Độ dập tắt dao động) sao cho robot có thể đứng vững ít nhất 2-3 giây trước khi sụp xuống.

---

## Giai Đoạn 4: Training Pipeline (GPU & Codebase)

- [ ] **1. Tối ưu theo Kiến trúc Card Đồ Họa (GPU Tweaks):**
  - Trên các dòng GPU cũ (như Pascal - Quadro P2000), MuJoCo Warp không tương thích với Newton solver mặc định.
  - Phải override option `solver` thành `cg` (Conjugate Gradient) để tránh lỗi biên dịch C++ (`libmathdx`).
- [ ] **2. Huấn luyện phân cấp (Curriculum Learning):**
  - Luôn khởi chạy Train trên `Mjlab-Velocity-Flat-HuD03` (địa hình phẳng) trước với tham số môi trường song song lớn nhất GPU có thể chịu được (VD: `--env.scene.num-envs 1024`).
  - Sau khi Loss hội tụ, mới chuyển qua task `Rough` (gồ ghề) bằng cách kế thừa file Checkpoint đã học được từ Flat.
