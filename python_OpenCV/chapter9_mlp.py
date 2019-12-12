from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train)

row, col = 28, 28
X_train = X_train.reshape(X_train.shape[0], row, col, 1)

print(X_train.shape)