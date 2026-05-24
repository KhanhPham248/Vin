# Phân Tích Chi Tiết Tham Số Huấn Luyện HU-D03: Cơ Sở Lý Thuyết & Ý Nghĩa Vật Lý

Tài liệu này giải thích chi tiết toàn bộ các tham số điều khiển (PD gains, Action Scale) và các tham số hàm thưởng (Rewards, Penalties) được áp dụng trong quá trình huấn luyện Locomotion (học đi) của robot Humanoid **LimX HU-D03** trên nền tảng `mjlab` (MuJoCo + NVIDIA Warp).

---

## 1. Thiết Kế Bộ Điều Khiển Cấp Thấp (Low-Level PD Gain Design)

Mô phỏng robot trong `mjlab` sử dụng bộ điều khiển vị trí khớp dạng PD (Proportional-Derivative) hoạt động ở cấp độ tần số cao của MuJoCo:
$$\tau = K_p (q^* - q) + K_d (\dot{q}^* - \dot{q})$$
Trong đó:
*   $\tau$: Torque (Mô-men xoắn) do động cơ sinh ra.
*   $q^*, \dot{q}^*$: Góc khớp mục tiêu (do Policy/Actor xuất ra) và vận tốc mục tiêu.
*   $q, \dot{q}$: Góc khớp thực tế và vận tốc khớp thực tế đo được từ cảm biến.
*   $K_p, K_d$: Hệ số khuếch đại vị trí (Stiffness - Độ cứng) và hệ số khuếch đại vận tốc (Damping - Độ giảm chấn).

### 📐 Công thức toán học & Hệ thống Dao động Bậc 2
Để khớp robot hoạt động như một **hệ dao động bậc hai cản tới hạn (critically damped second-order system)** (nhằm triệt tiêu tối đa hiện tượng rung lắc dao động nhưng vẫn bám mục tiêu nhanh nhất), $K_p$ và $K_d$ được thiết kế theo tần số tự nhiên (Natural Frequency - $\omega_n$) và hệ số cản (Damping Ratio - $\zeta$):

$$K_p = I_{armature} \cdot \omega_n^2$$
$$K_d = 2 \cdot \zeta \cdot I_{armature} \cdot \omega_n$$

Trong đó:
1.  **$I_{armature}$ (Mô-men quán tính phần ứng):** Đại diện cho quán tính cơ học của rotor động cơ sau khi phản chiếu qua hộp số. Các giá trị này được lấy trực tiếp từ file mô tả vật lý XML của robot:
    *   Hông & Gối: $0.15257125 \text{ kg}\cdot\text{m}^2$ (Khớp chịu lực nặng nhất).
    *   Achilles & Eo: $0.094889232 \text{ kg}\cdot\text{m}^2$.
    *   Vai & Khuỷu tay: $0.045760625 \text{ kg}\cdot\text{m}^2$.
2.  **$\omega_n$ (Natural Frequency - Tần số tự nhiên):** Được đặt là $10\text{ Hz}$ (tương đương $10 \times 2\pi \approx 62.83\text{ rad/s}$). Đây là tần số đáp ứng của khớp robot. Tần số này càng cao, khớp phản ứng càng nhanh và cứng, nhưng đòi hỏi lực động cơ lớn hơn.
3.  **$\zeta$ (Damping Ratio - Hệ số cản):** Được thiết kế là $2.0$ (Cản vượt hạn - Overdamped) để triệt tiêu hoàn toàn hiện tượng rung lắc/quá độ (overshoot) khi robot sải chân bước đi đột ngột.

---

## 2. Tỉ Lệ Hành Động (Action Scale)

Hành động đầu ra của mạng nơ-ron Actor được chuẩn hóa trong dải $[-1, 1]$. Góc khớp mục tiêu thực tế truyền xuống động cơ sẽ là:
$$q^* = q_{home} + a_t \cdot \text{Action Scale}$$
Trong đó:
*   $q_{home}$: Tư thế đứng mặc định (Keyframe).
*   $a_t$: Giá trị hành động do mạng nơ-ron sinh ra ($a_t \in [-1, 1]$).
*   $\text{Action Scale}$: Giới hạn góc quay tối đa mà Actor có thể dịch chuyển khỏi tư thế đứng trong một bước nhảy thời gian.

### ⚠️ Tại sao công thức tự động của G1 làm HU-D03 bị liệt khớp?
Công thức của G1 tính scale dựa trên giới hạn lực:
$$\text{Action Scale} = 0.25 \cdot \frac{\text{Effort Limit}}{K_p}$$
*   Đối với HU-D03, do khớp gối cực kỳ khỏe ($K_p \approx 602$), công thức này cho ra tỉ lệ hành động chỉ **$0.05\text{ rad}$ (tương đương $2.8^\circ$)**.
*   **Hậu quả:** Dù Actor có xuất ra hành động tối đa $1.0$, khớp gối robot chỉ di chuyển được vỏn vẹn $2.8^\circ$. Robot bị "khóa cứng chân", không thể co chân để sải bước, dẫn đến việc liên tục ngã và không học được dáng đi.

### 🛠️ Giải pháp điều chỉnh:
Chúng ta đã chuyển sang **cấu hình góc khớp thực tế trực tiếp**:
*   Khớp chân (Hông, Gối, Cổ chân Achilles, Eo): **$0.25\text{ rad}$ (tương đương $14.3^\circ$)**. Đảm bảo robot có biên độ cơ học đủ rộng để nhấc chân và sải bước mạnh mẽ.
*   Khớp tay và đầu: **$0.30\text{ rad}$ (tương đương $17.2^\circ$)**.

---

## 3. Hàm Phần Thưởng & Phạt (Rewards and Penalties)

Hàm thưởng tổng thể khuyến khích robot thực hiện nhiệm vụ di chuyển ổn định và phạt các hành vi phản tự nhiên.

### 🎯 3.1. Bám vận tốc tuyến tính (Track Linear Velocity)
Khuyến khích robot di chuyển bám sát vận tốc được yêu cầu bởi lệnh điều khiển (ví dụ đi thẳng $0.5\text{ m/s}$).
*   **Công thức:**
    $$R_{\text{lin}} = w \cdot \exp\left( -\frac{\|v_{xy} - v_{xy}^*\|^2}{2\sigma^2} \right)$$
    Trong đó:
    *   $w$: Trọng số thưởng (Đặt là `3.0`).
    *   $v_{xy}, v_{xy}^*$: Vận tốc mặt phẳng thực tế và vận tốc lệnh yêu cầu.
    *   $\sigma$ (Standard Deviation - Độ lệch chuẩn): Đóng vai trò quyết định **Cảnh quan Gradient (Gradient Landscape)**.

```
Điểm thưởng R
  1.0 |         * * *
      |       *       *      <-- std = 0.50 (Dễ bắt đầu, gradient rộng)
      |     *           *
      |   *               *
  0.0 |_*___________________*___ Sai số vận tốc (v - v*)
     -1.0        0.0        1.0
```

```
Điểm thưởng R
  1.0 |           *
      |          * *         <-- std = 0.15 (Cực hẹp, Gradient biến mất nếu sai số > 0.3)
      |        *     *
  0.0 |______*_________*________ Sai số vận tốc (v - v*)
     -1.0        0.0        1.0
```

*   **Ý nghĩa lý thuyết của $\sigma$:**
    *   Nếu $\sigma$ quá hẹp ($0.15$): Khi robot mới tập đi và có sai số vận tốc lớn (ví dụ $0.3\text{ m/s}$), điểm nhận được sẽ $\approx 0.13$ (gần như bằng 0). Do phần thưởng quá thưa thớt (Sparse Reward), mạng nơ-ron **không có tín hiệu Gradient** để cải thiện dáng đi.
    *   Nếu $\sigma$ quá rộng ($0.50$): Robot đứng im hoàn toàn vẫn ăn gian được rất nhiều điểm thưởng, dẫn đến nghiệm cục bộ (Local Optima) là đứng im lắc lư để bảo toàn tính mạng.
    *   **Giá trị tối ưu đã chỉnh:** **`0.25`**. Đây là điểm cân bằng lý tưởng giúp robot có lực đẩy gradient học đi từ những bước đầu tiên nhưng vẫn triệt tiêu rò rỉ điểm khi đã thuần thục.

### 🦶 3.2. Thời gian bay của chân (Air Time Reward)
Khuyến khích robot thực hiện động tác bước đi (gait) tuần hoàn bằng cách nhấc chân lên khỏi mặt đất thay vì trượt lết chân.
*   **Cơ sở lý thuyết:** Phần thưởng này chỉ kích hoạt khi chân robot nhấc lên không trung (vượt quá ngưỡng thời gian tối thiểu) và chỉ được trao **ngay tại thời điểm chân tiếp đất trở lại**.
    $$R_{\text{air}} = \sum_{i \in \{\text{feet}\}} (t_{\text{air}, i} - t_{\text{min}}) \cdot I(\text{tiếp đất})$$
*   **Vấn đề Hack Reward (Reward Hacking):** 
    Nếu đặt trọng số `air_time = 3.0` quá lớn, robot sẽ bỏ qua nhiệm vụ di chuyển bám vận tốc. Thay vào đó, nó chọn cách ngã ngửa ra sàn hoặc đứng yên co duỗi chân liên tục trong không trung để ăn điểm thưởng bay chân một cách dễ dàng.
*   **Giá trị tối ưu đã chỉnh:** Giảm từ `3.0` xuống **`1.0`** (chỉ bằng 1/3 thưởng vận tốc) để giữ vai trò là phần thưởng bổ trợ định hình dáng đi (Gait shaping), không đè lên phần thưởng nhiệm vụ chính.

### 📉 3.3. Phạt gia tốc hành động (Action Rate L2 Penalty)
Phạt sự thay đổi đột ngột giữa các hành động liên tiếp nhằm làm mịn dáng đi và bảo vệ động cơ.
*   **Công thức:**
    $$P_{\text{action}} = w_{\text{rate}} \cdot \|a_t - a_{t-1}\|^2$$
    Trong đó $w_{\text{rate}}$ là trọng số phạt (Đặt là `-0.05`).
*   **Cơ sở lý thuyết:** RSL-RL sử dụng mạng nơ-ron học sâu xuất góc khớp liên tục ở tần số $50\text{ Hz}$ (mỗi bước $0.02\text{ s}$). Nếu không có hình phạt này, Actor sẽ tự do thay đổi góc khớp cực kỳ đột ngột từ cực đại sang cực tiểu, gây ra hiện tượng **rung lắc giật cục tần số cao (flailing/shaking)**.
*   **Ý nghĩa vật lý:** Phạt L2 ép các hành động khớp trở nên trơn tru, mượt mà, giúp tăng độ bền cơ học động cơ thực tế và tiết kiệm điện năng cho robot.

### 📐 3.4. Giữ thân thẳng đứng (Upright Reward)
Đảm bảo phần thân trên (waist/torso) của robot luôn thẳng đứng so với trọng lực Trái Đất.
*   **Công thức:**
    $$R_{\text{upright}} = w_{\text{upright}} \cdot \exp\left( -\frac{1 - \mathbf{z}_{\text{body}} \cdot \mathbf{z}_{\text{world}}}{2\sigma_{\text{upright}}^2} \right)$$
*   **Cơ sở lý thuyết:** Tính tích vô hướng giữa trục $Z$ của thắt lưng robot (`waist_pitch_link`) và trục $Z$ tuyệt đối của thế giới. Tích này bằng $1.0$ khi robot hoàn toàn đứng thẳng đứng và giảm đi khi robot bị nghiêng hoặc ngã đổ. Trọng số đặt là `1.0` để ép robot giữ thăng bằng trọng tâm tuyệt đối.

---

## 4. Tóm Tắt Bản Đồ Tham Số Hàm Thưởng Đã Tối Ưu

| Tên tham số | Trọng số ($w$) | Độ lệch chuẩn ($\sigma$) | Ý nghĩa & Vai trò |
| :--- | :---: | :---: | :--- |
| `track_linear_velocity` | `+3.0` | `0.25` | Thưởng bám vận tốc tịnh tiến chỉ định (Nhiệm vụ chính). |
| `track_angular_velocity` | `+2.0` | `0.50` | Thưởng bám vận tốc quay góc chỉ định (Nhiệm vụ xoay hướng). |
| `upright` | `+1.0` | $\approx 0.45$ | Giữ thân trên luôn thẳng đứng, chống lật đổ. |
| `air_time` | `+1.0` | Ngưỡng `0.1s` | Khuyến khích nhấc chân sải bước, tạo dáng đi tuần hoàn tự nhiên. |
| `action_rate_l2` | `-0.05` | N/A | Phạt giật góc khớp đột ngột, triệt tiêu rung lắc tần số cao. |
| `soft_landing` | `-0.05` | N/A | Phạt lực dậm chân quá mạnh xuống sàn để bảo vệ cơ cấu bàn chân. |
| `self_collisions` | `-1.0` | N/A | Phạt va chạm giữa các bộ phận của robot (đặc biệt là hai gối đập vào nhau). |
