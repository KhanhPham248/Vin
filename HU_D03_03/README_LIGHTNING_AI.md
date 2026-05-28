# Cẩm nang Training HU-D03 ngầm trên Lightning AI (Sử dụng tmux)

Tài liệu này lưu lại toàn bộ các lệnh cần thiết để thiết lập môi trường, chạy huấn luyện (training) ngầm và cách để quá trình này không bị gián đoạn khi bạn tắt trình duyệt hoặc tắt máy tính.

---

## 1. Thiết lập lần đầu (Clone & Setup)

Mở Terminal trên Lightning AI và dán các lệnh sau:

```bash
# 1. Clone source code mới nhất từ GitHub
git clone https://github.com/KhanhPham248/Vin.git

# 2. Cài đặt uv (Trình quản lý package siêu tốc độ)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Chuyển vào thư mục chứa cấu hình pyproject
cd Vin/HU_D03_03
```

---

## 2. Các Task Training Có Sẵn

Có **2 task** được đăng ký (đều đã được tích hợp mặc định cấu hình dáng đi Unitree tối ưu nhất):

| Task ID | Địa hình | WandB Project | Mô tả |
|---|---|---|---|
| `Mjlab-Velocity-Flat-HuD03` | Flat (Phẳng) | `hu_d03_03` | Task chạy trên mặt phẳng, huấn luyện nhanh |
| `Mjlab-Velocity-Rough-HuD03` | Rough (Gồ ghề) | `hu_d03_03` | Task chạy trên địa hình phức tạp, gồ ghề |

> **Đặc điểm của phiên bản Unitree tối ưu:**
> - Thêm phase observation `[sin, cos]` (chu kỳ dáng đi) giúp robot phối hợp chuyển động chân mượt mà.
> - Cấu hình `foot_gait` cyclic reward (period `0.65`) giúp robot định hình bước đi đối xứng ổn định.
> - Tối ưu `stand_still` penalty bằng cách chỉ áp dụng lên các joint được kích hoạt (`ACTUATED_JOINT_NAMES`).

---

## 3. Bắt đầu phiên làm việc ngầm với `tmux`

Để tránh việc tiến trình bị ngắt do lỗi mạng hoặc đóng trình duyệt, chúng ta phải chạy lệnh trong một môi trường ảo có tên là `tmux`.

### Chạy Task Flat (Mặt phẳng)

```bash
# Tạo session tmux tên "hud03_flat"
tmux new -s hud03_flat

# Bên trong tmux — chạy Flat Task
uv run python scripts/train.py Mjlab-Velocity-Flat-HuD03 --env.scene.num-envs 8192
```

### Chạy Task Rough (Gồ ghề)

```bash
# Tạo session tmux tên "hud03_rough"
tmux new -s hud03_rough

# Bên trong tmux — chạy Rough Task
uv run python scripts/train.py Mjlab-Velocity-Rough-HuD03 --env.scene.num-envs 8192
```

> **Lưu ý số lượng môi trường (VRAM):**
> - GPU T4 (16GB): nên dùng `--env.scene.num-envs 4096` để đảm bảo ổn định không bị OOM.
> - GPU L4 (24GB) hoặc A10G: thoải mái dùng `--env.scene.num-envs 8192`.

---

## 4. Thoát ra ngoài (Để cho máy tự cày)

Khi thấy máy bắt đầu in ra số liệu Epoch/Metrics trên màn hình, bạn có thể "tách" (detach) khỏi màn hình này mà tiến trình vẫn tiếp tục chạy ngầm.

**Thao tác phím tắt:**
1. Nhấn giữ phím **`Ctrl`** và phím **`B`** (bấm một phát rồi thả tay ra).
2. Sau đó bấm nhẹ phím **`D`** (viết tắt của Detach).

Bây giờ bạn có thể tắt tab trình duyệt, tắt máy, đi ngủ. Server Lightning AI vẫn sẽ thay bạn train robot.

---

## 5. Cách vào lại để xem tiến độ

Bất cứ lúc nào bạn quay lại Lightning AI, muốn xem robot đã học được tới Epoch thứ bao nhiêu, chỉ cần mở Terminal lên và gõ:

```bash
tmux attach -t hud03_flat    # xem tiến trình Flat
tmux attach -t hud03_rough   # xem tiến trình Rough
```

Màn hình Terminal đang chạy dở hôm trước sẽ tự động hiện lại nguyên vẹn.

---

## 6. (Mẹo) Huỷ tiến trình

Nếu bạn muốn dừng huấn luyện để sửa đổi cấu hình hoặc làm lại từ đầu:
```bash
tmux kill-session -t hud03_flat
tmux kill-session -t hud03_rough
```
