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
cd Vin/HU_D03_locomotion
```

---

## 2. Các Task Training Có Sẵn

Có **4 task** được đăng ký, chia thành 2 loại chính:

| Task ID | Loại | Địa hình | WandB Project | Mô tả |
|---|---|---|---|---|
| `Mjlab-Velocity-Flat-HuD03` | **Standard** | Flat | `hu_d03_locomotion` | Task chính, bắt đầu tại đây |
| `Mjlab-Velocity-Flat-HuD03-Unitree` | **Unitree** | Flat | `hu_d03_locomotion_unitree` | Thêm phase gait + stand_still penalty |
| `Mjlab-Velocity-Rough-HuD03` | Standard | Rough | `hu_d03_locomotion` | Sau khi flat thành công |
| `Mjlab-Velocity-Rough-HuD03-Unitree` | Unitree | Rough | `hu_d03_locomotion_unitree` | Sau khi flat unitree thành công |

> **Sự khác biệt Standard vs Unitree:**
> - **Standard**: Dùng `feet_air_time_touchdown` reward (chỉ thưởng tại touchdown).
> - **Unitree**: Thêm phase observation `[sin, cos]` + `foot_gait` cyclic reward + `stand_still` penalty.

---

## 3. Bắt đầu phiên làm việc ngầm với `tmux`

Để tránh việc tiến trình bị ngắt do lỗi mạng hoặc đóng trình duyệt, chúng ta phải chạy lệnh trong một môi trường ảo có tên là `tmux`.

### Chạy 1 task (Standard Flat)

```bash
# Tạo session tmux tên "standard"
tmux new -s standard

# Bên trong tmux — chạy Standard Flat
uv run python scripts/train.py Mjlab-Velocity-Flat-HuD03 --env.scene.num-envs 8192
```

### Chạy 2 task song song (Standard + Unitree để so sánh)

```bash
# Session 1 — Standard Flat
tmux new -s standard
uv run python scripts/train.py Mjlab-Velocity-Flat-HuD03 --env.scene.num-envs 4096
# Detach: Ctrl+B → D

# Session 2 — Unitree Flat
tmux new -s unitree
uv run python scripts/train.py Mjlab-Velocity-Flat-HuD03-Unitree --env.scene.num-envs 4096
# Detach: Ctrl+B → D
```

> **Lưu ý VRAM khi chạy song song:**
> - T4 (16GB): dùng `--env.scene.num-envs 4096` cho mỗi task (tổng ~12-14GB).
> - L4 (24GB): có thể dùng `--env.scene.num-envs 8192` cho mỗi task.


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
tmux attach -t standard    # xem standard
tmux attach -t unitree     # xem unitree
```

Màn hình Terminal đang chạy dở hôm trước sẽ tự động hiện lại nguyên vẹn.

---

## 6. (Mẹo) Huỷ tiến trình nếu lỡ bị kẹt
Nếu bạn lỡ bấm linh tinh hoặc muốn hủy toàn bộ cái `tmux` tên là "training" này để làm lại từ đầu:
```bash
tmux kill-session -t standard
tmux kill-session -t unitree
```
