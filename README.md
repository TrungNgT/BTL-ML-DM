# SỬ DỤNG YOLOv8 ĐỂ NHẬN DIỆN GƯƠNG MẶT VÀ DỰ ĐOÁN CẢM XÚC

## Cài đặt
Cài đặt thư viện [ultralytics](https://github.com/ultralytics/ultralytics) bằng pip:
```bash
# Install ultralytics with pip:
pip install ultralytics
```
Trong trường hợp đã thử cài thư viện [ultralytics](https://github.com/ultralytics/ultralytics) mà khi chạy các chương trình vẫn báo lỗi thì có thể cài đặt theo cấu hình yêu cầu sau:
```bash
# Install with requirements.txt:
pip install requirements.txt
```

## Đối với mục đích sử dụng chương trình:
1. Đầu tiên vào file [choose_model](programs/choose_model.py) để lựa chọn các mô hình để sử dụng trong chương trình.


- Đối với mô hình để nhận diện khuôn mặt, có thể lấy đường dẫn ở thư mục [detect](models/detect)
```bash
detector_model = YOLO("path_to_detector_model")
```

- Đối với mô hình để phân loại cảm xúc, có thể lấy đường dẫn ở thư mục [classify](models/classify)
```bash
classifier_model = YOLO("path_to_classifier_model")
```

---------------------------------------------------------
2. Sử dụng các chương trình

Vào thư mục [programs](programs), sẽ thấy có 3 lựa chọn sử dụng chương trình:

- [work_on_images](programs/work_on_images.py) sẽ sử dụng để nhận diện và dự đoán biểu cảm trên các ảnh tĩnh. Có thể lấy ảnh từ tập [future_test](future_test) để sử dụng, đưa đường dẫn của ảnh tới:
```bash
image = cv2.imread("path_to_the_image")
```

- [work_on_video](programs/work_on_video.py) sẽ sử dụng để nhận diện và dự đoán biểu cảm các nhân vật xuất hiện trong 1 video .MP4 đầu vào. Các kết quả sẽ được hiển thị ở video đầu ra tương ứng. Có thể lấy file .MP4 từ tập [future_test](future_test) để sử dụng, đưa đường dẫn file tới:
```bash
video_path = os.path.join(VIDEOS_DIR, "path_to_the_video")
```

- [work_on_webcam](programs/work_on_webcam.py) sẽ sử dụng webcam trực tiếp của máy tính để thực hiện nhận diện mặt và phán đoán cảm xúc bằng camera thực. Để sử dụng chỉ cần run file [work_on_webcam](programs/work_on_webcam.py)


## Đối với mục đích huấn luyện thêm mô hình
- Có thể xem thông tin đánh giá, dữ liệu và lựa chọn pretrained model ở phần [train_process](train_process) và [data_preparation](data_preparation)
- Đối với nhiệm vụ phân loại (classification task), khuyến nghị chạy file [train](train_process\code) ở môi trường Colab
<a target="_blank" href="https://colab.research.google.com/drive/14QfCaIClnfSmHjjVkMNoMtZ0MlRhCwr6?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Đối với nhiệm vụ nhận diện gương mặt (face detection task) thì yêu cầu cài đặt thêm: 
