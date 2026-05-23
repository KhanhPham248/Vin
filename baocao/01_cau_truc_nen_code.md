# Cấu Trúc Nền Code — mjlab Framework

Về mặt kiến trúc tổng thể, **mjlab** được xây dựng dựa trên sự kết hợp của hai nền tảng lõi:
1. **Manager-based API (từ Isaac Lab):** Chia nhỏ và module hóa các thành phần của môi trường RL (Observation, Action, Reward...).
2. **MuJoCo Warp (từ DeepMind):** Xử lý mô phỏng vật lý trên không gian tensor của GPU để chạy song song hàng nghìn môi trường.

Để đáp ứng hai nền tảng này, cấu trúc code trong thư mục nguồn (`src/mjlab`) được thiết kế cực kỳ rành mạch và chia thành **4 khối (blocks) kiến trúc chính**:

---

### Khối 1: Lõi Mô Phỏng Vật Lý (Simulation & Physics)
*Nhiệm vụ: Cầu nối giữa mô tả vật lý (XML/MJCF) và mô phỏng tensor trên GPU.*

*   **`sim/`**: Trái tim của engine, bọc (wrap) MuJoCo Warp, quản lý bộ nhớ GPU và tối ưu hóa hiệu năng bằng công nghệ CUDA Graphs.
*   **`scene/`**: Nơi lắp ráp thế giới mô phỏng. Ghép nối nhiều đối tượng thành một thiết kế môi trường (`MjSpec`) duy nhất.
*   **`entity/`**: Lớp trừu tượng hóa cho đối tượng vật lý (Robot, Vật thể, Chướng ngại vật).
*   **`terrains/`**: Hệ thống sinh địa hình tự động (mặt phẳng, dốc, bậc thang...).
*   **`actuator/` & `sensor/`**: Các thành phần tương tác vật lý (động cơ, cảm biến) được thiết kế đặc thù để xử lý tính toán dạng mảng (batched) trên GPU.

### Khối 2: Logic Học Tăng Cường (Manager-Based MDP)
*Nhiệm vụ: Module hóa quá trình tương tác giữa Agent (Robot) và Môi trường theo dạng Markov Decision Process (MDP).*

*   **`envs/`**: Chứa môi trường chính (`ManagerBasedRlEnv`). Nơi định nghĩa vòng lặp môi trường, quản lý các bước nhảy vật lý và gọi các Manager.
*   **`managers/`**: Cốt lõi của thiết kế "Manager-based". Chứa 9 module quản lý độc lập (Action, Observation, Reward, Termination, Event, Command, Curriculum, Metrics, Recorder). Nhờ cấu trúc này, việc thêm/bớt reward hay observation chỉ đơn giản là tinh chỉnh file cấu hình mà không cần sửa code cốt lõi.

### Khối 3: Ứng Dụng & Thuật Toán (Application & RL)
*Nhiệm vụ: Áp dụng framework để giải quyết các bài toán robotics cụ thể.*

*   **`tasks/`**: Chứa các "Gói bài toán" (ví dụ: `velocity`, `tracking`). Đây là nơi người dùng thực sự làm việc: định nghĩa cấu hình bài toán cho các robot cụ thể (ví dụ dùng robot Unitree G1, chạy trên địa hình nào, hàm thưởng phạt ra sao).
*   **`rl/`**: Lớp Wrapper tích hợp thuật toán Học tăng cường (PPO) từ thư viện `rsl-rl`, chịu trách nhiệm xử lý quá trình huấn luyện (training).

### Khối 4: Công Cụ Hỗ Trợ (Tooling)
*Nhiệm vụ: Cung cấp giao diện chạy (CLI), hiển thị đồ họa và tài nguyên.*

*   **`scripts/`**: Chứa các mã lệnh thực thi chương trình qua Terminal (lệnh train, play, demo).
*   **`viewer/`**: Chịu trách nhiệm Render 3D và hiển thị trực quan thông tin (dùng Viser hoặc MuJoCo native viewer).
*   **`asset_zoo/`**: Kho lưu trữ các file mô hình 3D và XML (MJCF) của các loại robot có sẵn.
*   **`utils/`**: Các hàm và công cụ tiện ích phụ trợ (xử lý toán học, bộ đệm dữ liệu, gỡ lỗi NaN).
