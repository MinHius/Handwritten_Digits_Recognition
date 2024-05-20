import tensorflow as tf

# Lấy bộ dữ liệu mnist và cache.
def get_mnist_data():
    
    # Lấy bộ dữ liệu mnist. 
    path = 'mnist.npz'

    # Cache.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    return x_train, y_train, x_test, y_test

# Huấn luyện mô hình với bộ dữ liệu mnist. 
def train_model(x_train, y_train, x_test, y_test):
    
    # Lặp cho tới khi độ chính xác > 99%.
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy, stopping training!")
                self.model.stop_training = True    
    callbacks = myCallback()

    # Chuẩn hóa dữ liệu.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Tạo mô hình CNN với lớp Flatten và 2 lớp Dense.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    # Fit mô hình.
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
 
    print(history.epoch, history.history['accuracy'][-1])
    return model

# Hàm main.
def main():
    model = None
    
    # Huấn luyện model.
    print("Getting mnist data...")
    (x_train, y_train, x_test, y_test) = get_mnist_data()
    print("Training model...")
    model = train_model(x_train, y_train, x_test, y_test)
    print("Saving model...")
    model.save('model.h5')

# Khởi chạy.
if __name__ == '__main__':
    main()






