import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input


(X_train, y_train), (X_test, y_test) = mnist.load_data()


print("Training Images Shape :", X_train.shape)
print("Training Labels Shape :", y_train.shape)

print("Testing Images Shape :", X_test.shape)
print("Testing Labels Shape :", y_test.shape)


# DISPLAY SAMPLE IMAGES

fig1 = plt.figure(figsize=(10,5))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(X_train[i], cmap='gray')

    plt.title(f"Digit : {y_train[i]}")

    plt.axis('off')

plt.tight_layout()


# NORMALIZATION

X_train = X_train / 255.0
X_test = X_test / 255.0


# CREATE MODEL

model = Sequential([

    Input(shape=(28,28)),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),

    Dense(10, activation='softmax')

])


# MODEL SUMMARY

model.summary()


# COMPILE MODEL

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)


# TRAIN MODEL

history = model.fit(

    X_train,
    y_train,

    epochs=10,

    validation_data=(X_test, y_test)

)


# EVALUATE MODEL

loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss :", loss)

print("Test Accuracy :", accuracy)


# ACCURACY GRAPH

fig2 = plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['Training Accuracy', 'Validation Accuracy'])

plt.grid(True)


# PREDICTION

prediction = model.predict(X_test)

predicted_digit = np.argmax(prediction[0])

print("\nPredicted Digit :", predicted_digit)

print("Actual Digit :", y_test[0])


# SHOW PREDICTED IMAGE

fig3 = plt.figure(figsize=(4,4))

plt.imshow(X_test[0], cmap='gray')

plt.title(f"Predicted : {predicted_digit}")

plt.axis('off')


# SHOW ALL FIGURES TOGETHER

plt.show()