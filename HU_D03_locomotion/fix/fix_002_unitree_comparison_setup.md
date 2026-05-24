# Nhật Ký Sửa Đổi #002: Tích Hợp Hệ Thống Huấn Luyện Unitree-Style (Đối Chiếu Độc Lập)

**Ngày sửa đổi:** 24/05/2026
**Mục tiêu:** Tích hợp các tính năng cao cấp từ Unitree RL (chu kỳ pha `phase`, thưởng đồng bộ chân `feet_gait`, khóa dáng đứng im `stand_still`) vào một bộ huấn luyện riêng biệt để so sánh trực quan hiệu năng và chất lượng dáng đi của robot **HU_D03** mà không ảnh hưởng tới code gốc.

---

## 1. Cơ Chế Tách Biệt Độc Lập (Safety & Modular Isolation)
Để giữ an toàn tuyệt đối cho bộ huấn luyện cũ của bạn, tôi không viết đè hay sửa đổi cấu hình gốc. Thay vào đó, tôi triển khai cơ chế **kế thừa động (dynamic inheritance)**:
*   Môi trường Unitree mới sẽ nhập (import) trực tiếp cấu hình gốc rồi tự động chèn thêm (layering) các quan sát pha và hàm thưởng mới.
*   Nếu cấu hình gốc được bạn cải tiến (ví dụ: đổi PD gains, đổi tư thế đứng mặc định), bộ huấn luyện Unitree-style sẽ tự động thừa hưởng mà không cần sửa code lần hai!

---

## 2. Các File Mới Được Tạo

### 📁 `src/hu_d03_locomotion/tasks/mdp_unitree.py` [NEW]
*   Chứa code triển khai thuần bằng PyTorch các hàm của Unitree (không cần cài thêm thư viện hay sửa lõi `mjlab`):
    1.  `phase(...)`: Quan sát pha thời gian thực dưới dạng $[\sin(2\pi t/T), \cos(2\pi t/T)]$ với chu kỳ $T=0.6s$. Tự động tắt ($=0$) khi vận tốc lệnh gần bằng $0$.
    2.  `feet_gait(...)`: Thưởng đồng bộ chu kỳ chạm đất lệch pha $180^\circ$ giữa 2 chân.
    3.  `stand_still(...)`: Phạt lệch khớp khỏi tư thế tĩnh chỉ khi robot nhận lệnh đứng yên.

### 📁 `src/hu_d03_locomotion/tasks/velocity_unitree_env_cfg.py` [NEW]
*   Kế thừa cấu hình flat/rough cũ.
*   Tích hợp `phase` vào `actor` và `critic` observations.
*   Thay thế thưởng `air_time` truyền thống bằng `foot_gait` và thêm `stand_still`.

### 📁 `src/hu_d03_locomotion/tasks/rl_unitree_cfg.py` [NEW]
*   Kế thừa PPO runner cũ nhưng đổi `experiment_name` thành `"hu_d03_flat_unitree"` và `"hu_d03_rough_unitree"`.
*   **Phân tách dự án trên Weights & Biases (WandB):**
    *   **Bộ chạy gốc (Standard)**: Được gán riêng về dự án `wandb_project = "hu_d03_locomotion"` (thay vì chung đống `"mjlab"` mặc định).
    *   **Bộ chạy mới (Unitree-Style)**: Được gán hẳn sang dự án `wandb_project = "hu_d03_locomotion_unitree"`.
    *   *Lợi ích*: Cực kỳ sạch sẽ, dữ liệu log và đường đồ thị của 2 phương pháp được gom vào 2 Project khác nhau trên trang chủ WandB của bạn, không lo bị đè đè chồng chéo tên run!

---

## 3. Đăng Ký Task ID Mới (`tasks/__init__.py` [UPDATED])
Tôi đã đăng ký thành công hai Task mới vào hệ thống:
1.  **`Mjlab-Velocity-Flat-HuD03-Unitree`** (Môi trường phẳng)
2.  **`Mjlab-Velocity-Rough-HuD03-Unitree`** (Môi trường gồ ghề)

---

## 4. Hướng Dẫn Chạy & So Sánh (Usage Guide)

### Chạy trực quan (Play/Visualizer Mode)
Bạn có thể chạy thử trực quan để kiểm tra xem robot di chuyển thế nào với bộ cấu hình mới (chạy zero agent đứng im):
```bash
# Kiểm tra Flat Unitree
uv run python scripts/play.py Mjlab-Velocity-Flat-HuD03-Unitree --agent zero

# Kiểm tra Rough Unitree
uv run python scripts/play.py Mjlab-Velocity-Rough-HuD03-Unitree --agent zero
```

### Chạy huấn luyện (Training Mode)
Bắt đầu huấn luyện bộ não mới trên môi trường phẳng phẳng ngầm với `tmux`:
```bash
# Tạo session mới
tmux new -s train_hud03_unitree

# Bắt đầu chạy
uv run python scripts/train.py Mjlab-Velocity-Flat-HuD03-Unitree --env.scene.num-envs 8192
```

### So sánh kết quả trên TensorBoard
Trong khi chạy hoặc sau khi chạy xong, mở TensorBoard để so sánh dáng đi, độ mượt hành động, và tốc độ hội tụ của bộ não gốc vs bộ não Unitree-style:
```bash
tensorboard --logdir logs/
```
*(Trên giao diện TensorBoard, bạn sẽ thấy hai đường đồ thị màu sắc khác nhau của `hu_d03_flat` và `hu_d03_flat_unitree` hiển thị song song).*
