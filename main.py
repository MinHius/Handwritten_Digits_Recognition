import cv2
import numpy as np
import tensorflow as tf

# Define custom softmax_v2 function
def softmax_v2(x):
    return tf.nn.softmax(x)

# Dự đoán chữ số trong ảnh thu được.
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# Phần code của OpenCV.
startInference = False

# Nếu click vào màn hình thì ứng dụng sẽ bắt đầu lấy ảnh.
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

# Mở webcam và capture ảnh.
def start_cv(model):
    global startInference
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)

    background = np.zeros((480, 640), np.uint8)
    frameCount = 0
    
    # Threshold.
    threshold = 150  

    while True:
        ret, frame = cap.read()

        if startInference:
            frameCount += 1

            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

            resizedFrame = thr[240-75:240+75, 320-75:320+75]
            background[240-75:240+75, 320-75:320+75] = resizedFrame

            iconImg = cv2.resize(resizedFrame, (28, 28))
            
            # Dự đoán ảnh vừa thu được.
            res = predict(model, iconImg)

            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320-80, 240-80), (320+80, 240+80), (255, 255, 255), thickness=3)
            
            cv2.imshow('background', background)
        else:
            cv2.imshow('background', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run():
    # Load the model with custom objects
    model = tf.keras.models.load_model('model.h5', custom_objects={'softmax_v2': softmax_v2})
    print('Loaded saved model.')
    print(model.summary())  
    start_cv(model)
    
run()
