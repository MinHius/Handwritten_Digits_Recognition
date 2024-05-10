import cv2
import numpy as np
import tensorflow as tf

# get mnist data and cache it 
def get_mnist_data():
    # get mnist data 
    path = 'mnist.npz'

    # get data - this will be cached 
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    return x_train, y_train, x_test, y_test

# train model with mnist data 
def train_model(x_train, y_train, x_test, y_test):
    # set up TF model and train 
    # callback 
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy, stopping training!")
                self.model.stop_training = True
                
    callbacks = myCallback()

    # normalize 
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # create model 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    # fit model
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    # stats 
    print(history.epoch, history.history['accuracy'][-1])
    return model

# predict digit using image passed in 
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# opencv part 
startInference = False

def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

def start_cv(model):
    global startInference
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)

    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    threshold = 150  # Fixed threshold value

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

def main():
    model = None
    try:
        model = tf.keras.models.load_model('model.h5')
        print('Loaded saved model.')
        print(model.summary())
    except:
        print("Getting mnist data...")
        (x_train, y_train, x_test, y_test) = get_mnist_data()
        print("Training model...")
        model = train_model(x_train, y_train, x_test, y_test)
        print("Saving model...")
        model.save('model.h5')
    
    print("Starting cv...")
    start_cv(model)

if __name__ == '__main__':
    main()






