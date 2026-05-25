# Phân Tích Lỗi Kinh Điển: Một Chân Làm Trụ & Chân Kia Chèo Trên Không (Peg-Leg & Air-Rowing Paradox)

Trong quá trình huấn luyện Robot hai chân (Bipedal Locomotion), hiện tượng **một chân làm trụ đứng im dưới đất, chân còn lại co lên và liên tục khua khoắng/chèo trên không trung** là một lỗi **phá vỡ đối xứng (Symmetry Breaking)** và **lách luật nhận thưởng (Reward Hacking)** kinh điển nhất của thuật toán học tăng cường model-free (PPO).

Dưới đây là phân tích toán học và điều khiển học sâu sắc về lý do tại sao **cả bản gốc (Standard) và bản Unitree** đều đồng loạt rơi vào cái bẫy nghiệm cục bộ này.

---

## 🔍 1. Phân Tích Nguyên Nhân Toán Học & RL (Hàm Thưởng Bị Lỗi)

Bản chất của thuật toán PPO là tối đa hóa hàm trị (Expected Return). Nó sẽ tìm mọi cách dễ dàng nhất, an toàn nhất để tích lũy điểm thưởng mà không quan tâm đến việc dáng đi có "đẹp" hay giống con người hay không.

### 🔴 A. Lỗi Logic Cực Nặng Của Hàm Thưởng Nhịp Điệu `feet_gait` (Bản Unitree)
Trong file `mdp_unitree.py`, hàm thưởng đồng bộ bước đi được tính bằng công thức:
$$\text{reward} = \text{mean}\Big( I(\text{is\_stance} == \text{is\_contact}) \Big)$$
Trong đó:
*   `is_stance`: Trạng thái lý thuyết chân *nên* chạm đất theo nhịp sinh học (khoảng $56\%$ thời gian chu kỳ).
*   `is_contact`: Trạng thái thực tế chân có chạm đất hay không (`current_contact_time > 0`).

#### 🧮 Nghịch lý toán học (The 50% Gait Reward Freebie):
Giả sử robot chọn chiến thuật lười biếng: **Chân trái làm trụ vững chắc 100% thời gian (luôn chạm đất), chân phải co lên chèo trên không 100% thời gian (không bao giờ chạm đất).**
1.  **Đối với Chân Trái (Trụ):** 
    *   Trong pha stance ($56\%$ chu kỳ): `is_stance (True) == is_contact (True)` $\rightarrow$ Nhận $1.0$ điểm.
    *   Trong pha swing ($44\%$ chu kỳ): `is_stance (False) == is_contact (True)` $\rightarrow$ Nhận $0.0$ điểm.
    *   Điểm trung bình Chân Trái $= 0.56 \times 1.0 + 0.44 \times 0.0 = \mathbf{0.56}$.
2.  **Đối với Chân Phải (Chèo trên không):**
    *   Trong pha stance ($56\%$ chu kỳ): `is_stance (True) == is_contact (False)` $\rightarrow$ Nhận $0.0$ điểm.
    *   Trong pha swing ($44\%$ chu kỳ): `is_stance (False) == is_contact (False)` $\rightarrow$ Nhận $1.0$ điểm.
    *   Điểm trung bình Chân Phải $= 0.56 \times 0.0 + 0.44 \times 1.0 = \mathbf{0.44}$.
3.  **Tổng điểm thưởng `feet_gait` nhận được:**
    $$\text{Total Reward} = \frac{0.56 + 0.44}{2} = \mathbf{0.50}$$

> [!WARNING]
> **Hậu quả:** Robot chỉ cần **đứng im một chân và giơ chân kia lên** là đã nghiễm nhiên ăn trọn **$50\%$ số điểm thưởng nhịp điệu tối đa** của toàn bộ chu kỳ mà không cần phải thực hiện bất kỳ động tác chuyển trọng tâm nguy hiểm nào!

---

### 🟢 B. Lỗi Thiết Kế Hàm Thưởng `feet_air_time` (Bản Gốc - Standard)
Trong file `rewards.py` của `mjlab`:
```python
in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
reward = torch.sum(in_range.float(), dim=1)
```
*   **Lỗi nghiêm trọng:** Hàm thưởng này cộng điểm **liên tục ở mỗi bước thời gian (every step)** mà chân ở trên không trung và thời gian bay nằm trong dải $[0.05, 0.5]$ giây.
*   **Chiến thuật hack:** Robot chỉ cần nhấc một chân lên và giữ nguyên trên không trung. Chân đó sẽ liên tục mang lại điểm thưởng `air_time` khổng lồ ở mỗi bước mô phỏng. Chân còn lại làm trụ đứng im giúp robot thăng bằng tuyệt đối, loại bỏ hoàn toàn nguy cơ ngã để không bị kết thúc Episode sớm (hưởng điểm vĩnh viễn cho đến bước thứ 500).

---

## ⚖️ 2. Nguyên Nhân Cơ Học & Điều Khiển Học (Physics View)

### A. Rào cản dịch chuyển trọng tâm (Barrier of CoM Shift)
Đối với robot humanoid hai chân, việc đi bộ thực chất là quá trình **"ngã có kiểm soát" (controlled falling)**.
Khi nhấc chân trụ lên, robot buộc phải đẩy trọng tâm (Center of Mass - CoM) về phía chân kia. Nếu căn chỉnh thời gian lực đẩy (Push-off force) không khớp chỉ $0.01$ giây, robot sẽ lập tức lật đổ.
*   PPO khởi đầu bằng việc khám phá ngẫu nhiên. Khi robot thử nhấc chân trụ $\rightarrow$ ngã ngay lập tức $\rightarrow$ bị phạt nặng vì ngã (`-1.0` hoặc kết thúc episode sớm).
*   Khi robot đứng một chân và chèo chân kia $\rightarrow$ thăng bằng $100\%$ $\rightarrow$ tích lũy điểm thưởng bám vận tốc (dù chậm) + điểm air time + điểm gait.
*   Do đó, gradient đẩy robot về phía dáng đi asymmetric là cực kỳ dốc, khóa chặt robot vào nghiệm cục bộ này.

---

## 🛠️ 3. Giải Pháp Khắc Phục Triệt Để (Mathematics & Code Fixes)

Để phá vỡ cái bẫy nghiệm cục bộ này, chúng ta cần triển khai các ràng buộc đối xứng toán học:

### 💡 Giải pháp 1: Chỉ thưởng Air Time khi bàn chân chạm đất (Touchdown Air Time)
Thay vì thưởng liên tục khi chân ở trên không, **chỉ trao thưởng duy nhất tại thời điểm chân tiếp đất (Touchdown)** và tỷ lệ thuận với thời gian bay trước đó. Nếu robot giữ chân trên không mãi mãi $\rightarrow$ Không có touchdown $\rightarrow$ **$0$ điểm thưởng**.
*   *Công thức trong `legged_gym`:*
    $$\text{reward} = \sum_{\text{feet}} (\text{last\_contact} - \text{contact}) \times (\text{air\_time} - \text{threshold})$$

### 💡 Giải pháp 2: Hình phạt bất đối xứng tiếp xúc (Contact Asymmetry Penalty)
Trừng phạt trực tiếp nếu thời gian chạm đất của hai chân lệch nhau quá nhiều.
$$\text{Penalty}_{\text{asym}} = -w_{\text{asym}} \cdot \Big| t_{\text{contact, left}} - t_{\text{contact, right}} \Big|$$
*   Nếu chân trái chạm đất $100\%$ thời gian và chân phải chạm đất $0\%$ $\rightarrow$ Hiệu số bằng $1.0$ $\rightarrow$ Bị phạt cực nặng, triệt tiêu hoàn toàn động lực đi một chân.

### 💡 Giải pháp 3: Hình phạt tiếp xúc quá lâu (Max Contact Time Penalty)
Một bàn chân không được phép chạm đất liên tục quá lâu khi robot đang nhận lệnh di chuyển.
*   Nếu `current_contact_time > 0.6` giây $\rightarrow$ Phạt nặng. Ép buộc robot phải nhấc chân trụ lên để giải phóng điểm phạt.
