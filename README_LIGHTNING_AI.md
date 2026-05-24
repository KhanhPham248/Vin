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

## 2. Bắt đầu phiên làm việc ngầm với `tmux`

Để tránh việc tiến trình bị ngắt do lỗi mạng hoặc đóng trình duyệt, chúng ta phải chạy lệnh trong một môi trường ảo có tên là `tmux`.

```bash
# 1. Tạo một cửa sổ làm việc ngầm tên là "training"
tmux new -s training
```

Sau khi gõ lệnh trên, bạn sẽ thấy một thanh trạng thái màu xanh lá/vàng xuất hiện ở đáy màn hình. Bây giờ bạn đang ở bên trong `tmux`.

```bash
# 2. Bắt đầu training (uv sẽ tự động xử lý môi trường)
uv run python scripts/train.py Mjlab-Velocity-Flat-HuD03 --env.scene.num-envs 8192
```

---

## 3. Thoát ra ngoài (Để cho máy tự cày)

Khi thấy máy bắt đầu in ra số liệu Epoch/Metrics trên màn hình, bạn có thể "tách" (detach) khỏi màn hình này mà tiến trình vẫn tiếp tục chạy ngầm.

**Thao tác phím tắt:**
1. Nhấn giữ phím **`Ctrl`** và phím **`B`** (bấm một phát rồi thả tay ra).
2. Sau đó bấm nhẹ phím **`D`** (viết tắt của Detach).

Bây giờ bạn có thể tắt tab trình duyệt, tắt máy, đi ngủ. Server Lightning AI vẫn sẽ thay bạn train robot.

---

## 4. Cách vào lại để xem tiến độ

Bất cứ lúc nào bạn quay lại Lightning AI, muốn xem robot đã học được tới Epoch thứ bao nhiêu, chỉ cần mở Terminal lên và gõ:

```bash
tmux attach -t training
```

Màn hình Terminal đang chạy dở hôm trước sẽ tự động hiện lại nguyên vẹn.

---

## 5. (Mẹo) Huỷ tiến trình nếu lỡ bị kẹt
Nếu bạn lỡ bấm linh tinh hoặc muốn hủy toàn bộ cái `tmux` tên là "training" này để làm lại từ đầu:
```bash
tmux kill-session -t training
```
