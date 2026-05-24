# Hướng Dẫn Chi Tiết Đọc Đồ Thị WandB: Huấn Luyện RL Humanoid HU-D03

Khi chạy huấn luyện với `mjlab` + `RSL-RL`, Weights & Biases (WandB) sẽ ghi nhận hàng chục đồ thị thời gian thực. Việc hiểu rõ ý nghĩa của từng đồ thị giúp bạn nhanh chóng chẩn đoán sức khỏe của mạng nơ-ron, phát hiện lỗi phần cứng/hàm thưởng, và biết khi nào nên dừng huấn luyện (Early Stopping).

Đồ thị WandB được chia làm 4 nhóm chính dưới đây:

---

## Group 1: Chỉ Số Sức Khỏe Thuật Toán PPO (Brain/Optimization Metrics)

Nhóm đồ thị này phản ánh trực tiếp trạng thái tối ưu hóa của mạng nơ-ron Actor (Chính sách - Policy) và Critic (Hàm giá trị - Value Function).

### 1. `Mean surrogate loss` (Surrogate Loss)
*   **Ý nghĩa:** Trọng tâm tổn thất của mạng Actor (Policy Gradient). Nó đánh giá xem chính sách mới tốt hơn chính sách cũ bao nhiêu, sau khi đã bị cắt clip (Clipping) để tránh thay đổi quá đột ngột.
*   **Kỳ vọng:** Dao động xung quanh 0 (thường trong dải $[-0.05, 0.05]$). 
*   **Chẩn đoán:** Nếu đồ thị này vọt lên dương quá lớn hoặc âm quá sâu rồi đứng im $\rightarrow$ Tốc độ học (Learning Rate) quá lớn làm chính sách bị đổ vỡ (Policy Collapse).

### 2. `Mean value loss` (Value Loss)
*   **Ý nghĩa:** Đánh giá sai số của mạng Critic khi dự đoán tổng lượng điểm thưởng tích lũy (Return) trong tương lai từ trạng thái hiện tại.
*   **Kỳ vọng:** Bắt đầu rất cao (do Critic chưa biết dự đoán), sau đó **giảm dần và hội tụ ổn định** về một giá trị nhỏ.
*   **Chẩn đoán:** Nếu Value Loss tăng vô hạn hoặc đột ngột nhảy vọt $\rightarrow$ Hàm thưởng của bạn đang bị nhiễu động dữ dội hoặc robot bị ngã liên tục ở các thế quái dị khiến Critic không thể học được quy luật.

### 3. `Mean entropy loss` (Entropy Loss)
*   **Ý nghĩa:** Đo lường **sự hỗn loạn/khả năng khám phá (Exploration)** của chính sách. Entropy càng âm (trị tuyệt đối càng lớn), chính sách càng ngẫu nhiên. Entropy tiến về 0, chính sách càng quyết định (Deterministic).
*   **Kỳ vọng:** Giảm dần (trở nên ít âm hơn) một cách từ từ suốt quá trình huấn luyện khi robot dần tìm ra dáng đi ổn định.
*   **Chẩn đoán:**
    *   *Entropy sụp đổ quá nhanh:* Robot bị kẹt vào nghiệm cục bộ (Local Optima) quá sớm (ví dụ đứng im hoặc ngã ngửa).
    *   *Entropy không giảm:* Robot không học được gì, chỉ flailing chân ngẫu nhiên.

### 4. `Mean action std` (Độ lệch chuẩn của hành động)
*   **Ý nghĩa:** Thể hiện biên độ khám phá của các khớp. 
*   **Kỳ vọng:** Giảm dần từ dải ban đầu ($1.0$ hoặc $0.8$) xuống dải thấp hơn ($0.2 \sim 0.3$), tương đương với việc Actor đã tự tin vào các quyết định góc khớp của mình.

---

## Group 2: Chỉ Số Hiệu Năng Tổng Thể (Performance & Survival)

### 1. `Mean reward` (Điểm thưởng trung bình mỗi Episode)
*   **Ý nghĩa:** Điểm tổng hợp thu được của toàn bộ hàm thưởng trong một chu kỳ (Episode). Đây là thước đo thành công tối thượng của bài toán RL.
*   **Kỳ vọng:** **Tăng liên tục theo dạng đường cong logarit** và đi ngang (hội tụ) ở giai đoạn cuối.

### 2. `Mean episode length` (Độ dài sống sót trung bình)
*   **Ý nghĩa:** Số bước chân (time steps) robot duy trì đứng vững trước khi bị ngã (bị kích hoạt điều kiện Termination).
*   **Kỳ vọng:** Tăng nhanh và chạm trần tối đa (mặc định là `500` hoặc `1000` bước). Khi đồ thị này đạt tối đa, nghĩa là robot đã học được cách **thăng bằng vĩnh viễn** mà không bao giờ ngã.

### 3. `SPS` (Steps Per Second - Tốc độ lấy mẫu)
*   **Ý nghĩa:** Số khung hình vật lý được tính toán và đẩy qua mạng nơ-ron mỗi giây.
*   **Kỳ vọng:** Đi ngang ổn định (với L4 thường đạt $35,000 \sim 50,000$ SPS). Nếu SPS sụt giảm đột ngột $\rightarrow$ Máy chủ đang bị quá nhiệt hoặc nghẽn phần cứng.

---

## Group 3: Chi Tiết Hàm Thưởng (Episode_Reward Breakdown)

Tất cả đồ thị có tiền tố `Episode_Reward/...` thể hiện điểm số trung bình của từng thành phần thưởng/phạt riêng lẻ. Đây là nơi bạn phát hiện ra robot đang làm tốt điều gì và đang bị phạt nặng ở đâu.

```
                  ┌─── Episode_Reward/track_linear_velocity ──> (Tốt: Tăng lên sát +3.0)
                  ├─── Episode_Reward/upright ───────────────> (Tốt: Giữ gần +1.0)
EPISODE_REWARDS  ├─── Episode_Reward/air_time ──────────────> (Tốt: Đạt từ +0.5 đến +1.0)
                  ├─── Episode_Reward/action_rate_l2 ────────> (Tốt: Tiến gần về 0, giảm phạt)
                  └─── Episode_Reward/self_collisions ───────> (Tốt: Bằng 0, không va chạm gối)
```

### 1. `Episode_Reward/track_linear_velocity`
*   **Kỳ vọng:** Tăng từ 0 lên sát giá trị trọng số cực đại (`+3.0`). Thể hiện robot đang đi đúng tốc độ chỉ định.

### 2. `Episode_Reward/upright`
*   **Kỳ vọng:** Duy trì ổn định ở mức sát `+1.0`. Nếu tụt sâu $\rightarrow$ Robot đang đi trong tư thế khom lưng hoặc bị nghiêng ngả vẹo vọ.

### 3. `Episode_Reward/air_time`
*   **Kỳ vọng:** Tăng dần từ 0 lên khoảng `0.5 ~ 1.0`. Cho thấy robot bắt đầu có nhịp điệu nhấc chân rõ ràng.

### 4. `Episode_Reward/action_rate_l2` (Phạt đổi góc đột ngột)
*   **Kỳ vọng:** Bắt đầu âm rất nặng (phạt lớn do robot flailing dữ dội), sau đó **giảm phạt dần và tiến sát về 0**. Cho thấy góc khớp di chuyển mượt mà, không giật cục.

### 5. `Episode_Reward/self_collisions`
*   **Kỳ vọng:** Bằng 0 tuyệt đối. Nếu đồ thị này có giá trị âm lớn $\rightarrow$ Hai đầu gối hoặc hai cổ chân của robot đang liên tục va đập chéo vào nhau gây hỏng hóc cơ học.

---

## Group 4: Chỉ Số Chẩn Đoán Dáng Đi Vật Lý (Metrics)

Nhóm đồ thị có tiền tố `Metrics/...` là các thông số chẩn đoán vật lý trích xuất trực tiếp từ mô phỏng MuJoCo, không tham gia vào hàm loss RL nhưng cực kỳ quan trọng để đánh giá chất lượng dáng đi.

### 1. `Metrics/air_time_mean` (Thời gian bay trung bình của chân)
*   **Ý nghĩa:** Thời gian trung bình mỗi bàn chân nhấc trên không trung trong một bước bước.
*   **Chỉ số chuẩn:** Đối với robot Humanoid đi bộ tốc độ trung bình, giá trị đẹp là **`0.2s` đến `0.4s`**. Nếu lớn hơn $\rightarrow$ Robot đang nhảy lò cò hoặc lò dò quá chậm. Nếu bằng 0 $\rightarrow$ Robot đang kéo lê chân dưới sàn.

### 2. `Metrics/slip_velocity_mean` (Vận tốc trượt bàn chân)
*   **Ý nghĩa:** Đo lường độ trượt của bàn chân trên mặt đất khi đang ở pha đứng (stance phase).
*   **Chỉ số chuẩn:** Càng sát 0 càng tốt. Chỉ số này cao nghĩa là robot đang đi trên "băng", bàn chân bị trượt lết gây mất mô-men lực đẩy.

### 3. `Metrics/landing_force_mean` (Lực va chạm khi tiếp đất)
*   **Ý nghĩa:** Lực phản đường cực đại khi chân chạm đất.
*   **Chỉ số chuẩn:** Nên ở mức vừa phải (dưới $200\text{ N}$). Lực quá lớn nghĩa là robot đang "nện" chân cực mạnh xuống sàn, dễ gây vỡ kết cấu bàn chân thực tế.

---

## 🛠️ Cẩm Nang Chẩn Đoán Lỗi Nhanh Qua Đồ Thị WandB

| Hiện Tượng Trên Đồ Thị | Nguyên Nhân Hệ Thống | Giải Pháp Khắc Phục |
| :--- | :--- | :--- |
| `Episode Length` kẹt ở mức thấp ($<100$), `Value Loss` tăng vô hạn. | Robot ngã ngay lập tức sau khi xuất phát do mất thăng bằng hoặc lực động cơ quá yếu. | Kiểm tra `stiffness` / `damping` của khớp eo và hông; giảm bớt vận tốc lệnh ban đầu của curriculum. |
| `track_linear_velocity` luôn bằng 0, nhưng `air_time` đạt điểm tối đa. | **Reward Hacking:** Robot bị liệt khớp chân hoặc lách luật bằng cách nằm ngửa đạp chân lên trời để ăn thưởng bay chân. | Kiểm tra `Action Scale` xem chân có bị liệt không; giảm trọng số `air_time` xuống `1.0` hoặc ít hơn. |
| `Entropy` sụp đổ về 0 quá nhanh chỉ sau 100 iterations. | **Exploration Failure:** Mạng nơ-ron bị kẹt cứng vào một tư thế tĩnh (thường là đứng im lắc lư) và không dám thử dáng đi mới. | Tăng hệ số `entropy_coef` trong `rl_cfg.py` lên `0.01` hoặc `0.02`; nới rộng `std` bám vận tốc. |
| `action_rate_l2` phạt cực kỳ nặng và không có xu hướng giảm. | Robot đang đi bằng cách rung giật tần số cao, các khớp bị co giật liên tục. | Tăng trọng số phạt `action_rate_l2` lên mức nặng hơn (ví dụ từ `-0.01` thành `-0.05` hoặc `-0.10`). |
| Đồ thị `self_collisions` liên tục báo điểm phạt âm sâu. | Hai chân robot quá sát nhau, đùi hoặc gối bị đập chéo vào nhau liên tục khi đi. | Bổ sung phạt va chạm tự thân nặng hơn; hoặc tăng nhẹ khoảng cách đứng mặc định của hai chân (`default_joint_positions`). |
