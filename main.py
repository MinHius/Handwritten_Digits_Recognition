import cv2
import numpy as np
import tensorflow as tf

# Hàm softmax biến 10 đầu ra từ mạng nơ-ron thành xác suất.
def softmax_v2(x):
    return tf.nn.softmax(x)

# Dự đoán chữ số trong ảnh thu được bằng hàm argmax - lấy xác suất cao nhất trong 10 xác suất.
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# Hàm xác định cửa sổ camera đã được click chưa.
startInference = False

# Nếu click vào màn hình thì ứng dụng sẽ bắt đầu lấy ảnh.
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

# Kiểm soát giá trị ngưỡng.
threshold = 100
def on_threshold(x):
    global threshold
    threshold = x

# Mở webcam và capture ảnh.
def start_cv(model):
    global threshold
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()

        if (startInference):
            
            # Đếm số khung hình hiện tại.
            frameCount += 1

            # Làm một khung màu đen để giới hạn khung hình.
            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Áp dụng giá trị ngưỡng.
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)
           

            # Trích xuất ảnh từ khung hình trung tâm.
            resizedFrame = thr[120:360, 200:440]
            background[120:360, 200:440] = resizedFrame

            # Thay đổi kích thước cho phù hợp mô hình.
            iconImg = cv2.resize(resizedFrame, (28, 28))
            
            # Dự đoán khung hình vừa lấy.
            res = predict(model, iconImg)

            # Mỗi 5 khung hình thì số ở góc sẽ được reset. 
            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            # In ra dự đoán ở góc màn hình.
            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (200, 120), (440, 360), (255, 255, 255), thickness=3)  # Adjusted the region of interest
            
            # Áp dụng khung giới hạn màu đen.
            cv2.imshow('background', background)
        else:
            # Chạy cam bình thường.
            cv2.imshow('background', frame)

        # Nếu phím 'q' được bấm, dừng ứng dụng.
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    # Tắt camera và đóng cửa sổ.
    cap.release()
    cv2.destroyAllWindows()




def run():
    # Gọi model với hàm softmax.
    model = tf.keras.models.load_model('model.h5', custom_objects={'softmax_v2': softmax_v2})
    print('Loaded saved model.')
    print(model.summary())  
    start_cv(model)
    
run()
