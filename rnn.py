import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import SimpleRNN, Dense

# Parameters
pitch = 20
step = 1
N = 100
n_train = int(N * 0.7)

# Generate data
def gen_data(x):
    return (x % pitch) / pitch

t = np.arange(1, N + 1)
y = np.array([gen_data(i) for i in t])

# Plot original data
plt.figure(figsize=(10, 5))
plt.plot(t, y, label="Original Data", color="blue")
plt.title("Original Data")
plt.show()

# Convert data to matrix
def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        Y.append(data[i + step])
    return np.array(X).reshape(-1, step, 1), np.array(Y)

# Split train and test sets
train, test = y[:n_train], y[n_train:]
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

print("Dimension (Before): ", train.shape, test.shape)
print("Dimension (After): ", x_train.shape, x_test.shape)

# Create and compile model
model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(step, 1), activation="relu"))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mse")
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# Plot training loss
plt.plot(hist.history['loss'])
plt.title("Model Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Predict and plot original vs predicted data
y_pred = model.predict(x_test).flatten()  # Flatten the prediction array
plt.figure(figsize=(10, 5))
plt.plot(t, y, label="Original", color="blue")
plt.plot(np.arange(n_train + step, N), y_pred, label="Predict", linestyle='dashed', color="red")
plt.axvline(x=n_train, color='magenta', linestyle='dotted')
plt.legend()
plt.title("Original vs Predict")
plt.xlabel("Time Steps")
plt.ylabel("Values")
plt.show()


