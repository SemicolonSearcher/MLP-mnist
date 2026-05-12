import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
import time

# load dataset
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

# filter cat and dog
trainfilter = (ytrain == 3) | (ytrain == 5)
testfilter = (ytest == 3) | (ytest == 5)

xtrain = xtrain[trainfilter.reshape(-1)]
ytrain = ytrain[trainfilter.reshape(-1)]

xtest = xtest[testfilter.reshape(-1)]
ytest = ytest[testfilter.reshape(-1)]

# convert labels
ytrain = (ytrain == 5).astype(int)
ytest = (ytest == 5).astype(int)

# preprocessing
xtrain1 = xtrain / 255.0
xtest1 = xtest / 255.0

xtrain2 = tf.image.resize(xtrain, (96,96)) / 255.0
xtest2 = tf.image.resize(xtest, (96,96)) / 255.0

# cnn model
cnn = Sequential([

    Conv2D(32, (3,3), activation='relu',
           input_shape=(32,32,3)),

    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# compile cnn
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train cnn
print("Training CNN")

start1 = time.time()

history1 = cnn.fit(
    xtrain1,
    ytrain,
    epochs=5,
    validation_data=(xtest1, ytest),
    verbose=2
)

end1 = time.time()

# evaluate cnn
cnnloss, cnnacc = cnn.evaluate(xtest1, ytest)

cnntime = end1 - start1

print("CNN Accuracy :", cnnacc)
print("CNN Training Time :", cnntime)

# transfer learning model
basemodel = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

basemodel.trainable = False

transfer_model = Sequential([

    basemodel,

    GlobalAveragePooling2D(),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

# compile transfer model
transfer_model.compile(
    optimizer=Adam(0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train transfer model
print("Training Transfer Learning Model")

start2 = time.time()

history2 = transfer_model.fit(
    xtrain2,
    ytrain,
    epochs=5,
    validation_data=(xtest2, ytest),
    verbose=2
)

end2 = time.time()

# evaluate transfer model
tfloss, tfacc = transfer_model.evaluate(xtest2, ytest)

tftime = end2 - start2

print("Transfer Learning Accuracy :", tfacc)
print("Transfer Learning Time :", tftime)

# accuracy graph
plt.plot(history1.history['val_accuracy'])

plt.plot(history2.history['val_accuracy'])

plt.title("CNN vs Transfer Learning")

plt.xlabel("Epochs")

plt.ylabel("Validation Accuracy")

plt.legend([
    "CNN",
    "Transfer Learning"
])

plt.show()

# bar graph
methods = ["CNN", "Transfer Learning"]

accuracies = [cnnacc, tfacc]

plt.bar(methods, accuracies)

plt.ylabel("Accuracy")

plt.title("Accuracy Comparison")

plt.ylim(0,1)

plt.show()

# final comparison
print("CNN Accuracy :", cnnacc)

print("Transfer Learning Accuracy :", tfacc)

print("CNN Training Time :", cnntime)

print("Transfer Learning Time :", tftime)