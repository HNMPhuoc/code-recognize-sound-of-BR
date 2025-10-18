# audio_classifier Backend

Dự án này là một server backend được xây dựng trên nền tảng **Python**. Nó được thiết kế để cung cấp các API nhận diện âm thanh của chim và chuột.

---

## Clone repository
1.  Clone repository về máy tính của bạn:
    ```bash
    git clone https://github.com/HNMPhuoc/code-recognize-sound-of-BR.git
    ```
---

2.  Cài đặt môi trường

    Sử dụng terminal tạo máy ảo
    ```bash
    python -m venv venv
    ```
    Kích hoạt máy ảo venv
    ```bash
    .\venv\Scripts\activate
    ```

3. Tải thư viện vào máy ảo
    ```bash
    pip install -r requirements.txt
    ```
    
4. Chạy server
Sử dụng lệnh uvicorn 1:
    ```bash
    uvicorn audio_classifier.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    Hoặc dùng lệnh uvicorn 2:
    ```bash
    uvicorn audio_classifier.main:app --host 0.0.0.0 --port 8000
    ```
5. Truy cập
Sử dụng 2 link url dưới đây:
    - link web index.html: http://127.0.0.1:8000/
    - link swagger: http://127.0.0.1:8000/docs
6. Nguồn kiểm tra
    - Tiếng chim: https://xeno-canto.org/
    - Tiếng chuột: https://www.soundjay.com/animal/ hoặc https://freesound.org/search/?q=mouse+squeak
    - Các file kiểm tra âm thanh có sẵn trong file-test