import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Lấy bộ dữ liệu MNIST và cache.
def get_mnist_data():
    # Lấy bộ dữ liệu mnist
    path = 'mnist.npz'
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    
    # Chuẩn hóa dữ liệu
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    
    return x_train, y_train, x_test, y_test

# Huấn luyện mô hình với bộ dữ liệu mnist
def train_model(x_train, y_train, x_test, y_test):
    # Khởi tạo mô hình Logistic Regression
    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')

    # Huấn luyện mô hình
    model.fit(x_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(x_test)

    # Tính độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    return model

# Hàm main
def main():
    model = None
    
    # Lấy dữ liệu MNIST
    print("Getting mnist data...")
    x_train, y_train, x_test, y_test = get_mnist_data()
    
    # Huấn luyện model
    print("Training model...")
    model = train_model(x_train, y_train, x_test, y_test)
    
    # Lưu mô hình
    print("Saving model...")
    import joblib
    joblib.dump(model, 'model.pkl')

# Khởi chạy
if __name__ == '__main__':
    main()







