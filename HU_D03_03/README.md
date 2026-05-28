# HU_D03_03: Hệ Thống Huấn Luyện Học Tăng Cường (RL) Cho Robot Humanoid HU_D03

Dự án phát triển chính thức các chính sách điều khiển (Locomotion) và bắt chước chuyển động (Mimic) tối ưu cho dòng robot Humanoid 31 bậc tự do (31 DOF) **HU_D03**. Dự án được xây dựng và tối ưu hóa trên nền tảng **mjlab** hỗ trợ mô phỏng vật lý MuJoCo tốc độ cao song song hóa cực đại qua NVIDIA Warp (CUDA).

---

## 📌 Các Tính Năng Nổi Bật

* **Hỗ Trợ Tối Đa 31 DOF:** Cấu hình chuẩn khớp toàn thân bao gồm cả chân nâng cao (achilles joints), hông, eo (waist), tay (shoulder/elbow/wrist) và đầu.
* **Nhóm Tác Vụ Khoa Học:** 
  1. **Locomotion (Điều khiển di chuyển):** Học đi bộ đa hướng linh hoạt trên địa hình phẳng và địa hình gồ ghề phức tạp.
  2. **Mimic (Bắt chước chuyển động mẫu):** Học cách tái hiện mượt mà các chuỗi hành động mẫu phức tạp từ tệp dữ liệu chuyển động (như nhảy múa, chạy, cử chỉ).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
HU_D03_03/
├── assets/                     # Tài nguyên 3D meshes và file cấu hình XML của robot
│   ├── motions/                # Nơi lưu trữ dữ liệu chuyển động mẫu (.npz) cho Mimic
│   └── robots/hu_d03/          # File mô hình XML chính thức của robot HU_D03
├── configs/                    # Các file cấu hình hệ thống
├── scripts/                    # Scripts tiện ích chính
│   ├── train.py                # Script huấn luyện chính (Locomotion & Mimic)
│   ├── play.py                 # Script chạy chính sách đã huấn luyện kèm UI hiển thị
│   └── csv_to_npz.py           # Tiện ích chuyển đổi dữ liệu CSV -> NPZ
├── src/hu_d03_03/              # Mã nguồn lõi của gói dự án
│   ├── robots/                 # Hằng số, tỷ lệ mô-men xoắn, tư thế chuẩn ban đầu (Home keyframe)
│   └── tasks/                  # Định nghĩa môi trường RL
│       ├── velocity/           # Tác vụ đi bộ (Locomotion: Flat & Rough)
│       └── mimic/              # Tác vụ bắt chước chuyển động (Mimic Flat)
├── pyproject.toml              # Quản lý dependency và cấu hình cài đặt dự án dạng editable
└── README.md                   # Tài liệu hướng dẫn sử dụng chính thức
```

---

## 🚀 Hướng Dẫn Huấn Luyện

Đảm bảo bạn đã kích hoạt môi trường ảo (ví dụ: `conda activate Vin`).

### 1. Huấn luyện Locomotion (Đi bộ mặt phẳng)
Tác vụ đi bộ trên mặt phẳng phẳng (`plane`), giúp robot nhanh chóng hội tụ dáng đi cơ bản:
```bash
python scripts/train.py Mjlab-Velocity-Flat-HuD03
```

### 2. Huấn luyện Locomotion (Đi bộ địa hình gồ ghề)
Tác vụ nâng cao di chuyển trên địa hình gồ ghề phức tạp có sinh địa hình tự động tăng dần độ khó:
```bash
python scripts/train.py Mjlab-Velocity-Rough-HuD03
```

### 3. Huấn luyện Mimic (Bắt chước chuyển động mẫu)
Học tái hiện tệp chuyển động mục tiêu (cần tệp `hu_d03_motion.npz` tại `assets/motions/`):
```bash
python scripts/train.py Mjlab-Mimic-Flat-HuD03
```

*(Mẹo: Bạn có thể thêm tham số `--env.scene.num-envs 16` để kiểm tra chạy thử nhanh cấu hình với số môi trường nhỏ trước khi train thật).*

---

## 📊 Hệ Thống Log & Quản Lý Dữ Liệu

### Lưu trữ cục bộ (Local Logs)
Log huấn luyện được ghi tự động vào các thư mục riêng biệt tại:
* Locomotion phẳng: `logs/rsl_rl/hu_d03_03_flat/`
* Locomotion gồ ghề: `logs/rsl_rl/hu_d03_03_rough/`
* Bắt chước Mimic: `logs/rsl_rl/hu_d03_03_mimic/`

### Trực quan hóa (Weights & Biases)
Toàn bộ biểu đồ Reward, Loss, Entropy của cả 3 tác vụ được tự động đồng bộ thời gian thực lên WandB Project:
* **WandB Project Name:** `hu_d03_03`

---

## 🔄 Công Cụ Chuyển Đổi Dữ Liệu CSV sang NPZ cho Mimic

Để nạp tệp chuyển động CSV thô từ bên ngoài vào hệ thống huấn luyện Mimic, bạn cần chạy script chuyển đổi động học ngược để tính toán tọa độ thế giới của từng khớp xương:

```bash
python scripts/csv_to_npz.py --input-file <ĐƯỜNG_DẪN_TỚI_FILE_CSV> --output-name hu_d03_motion.npz
```

*Ví dụ:*
```bash
python scripts/csv_to_npz.py --input-file ../unitree_rl_mjlab-main/src/assets/motions/g1/dance1_subject2.csv --output-name hu_d03_motion.npz
```

> ⚠️ **Lưu ý:** File CSV đầu vào cần có đủ 38 cột tương ứng cho vị trí gốc, hướng Quaternion và góc xoay của 31 khớp robot HU_D03_03 theo đúng thứ tự.

