import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
#load
(xt,yt),(xts,yts)=cifar10.load_data()


#3 and 5 dogs and cats
xtf=(yt==3)|(yt==5)
xtsf=(yts==3)|(yts==5)


#reshape
xt=xt[xtf.reshape(-1)]
yt=yt[xtf.reshape(-1)]


xts=xts[xtsf.reshape(-1)]
yts=yts[xtsf.reshape(-1)]


#dataset
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(xt[i])
    plt.title(yt[i])
plt.show()


#convert lbels
yt=(yt==5).astype(int)
yts=(yts==5).astype(int)


#resixe images
xt=tf.image.resize(xt,(96,96))/255.0
xts=tf.image.resize(xts,(96,96))/255.0


#load pretrained
basemodel=MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)


#freeze laeyrrs
basemodel.trainable=False
model=Sequential([
    basemodel,
    GlobalAveragePooling2D(),
    Dense(64,activation="relu"),
    Dropout(0.5),
    Dense(1,activation="sigmoid")
])


model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history1=model.fit(
    xt,
    yt,
    epochs=5,
    validation_data=(xts,yts),
    verbose=2
)


loss1,acc1=model.evaluate(xts,yts)
print("accuracy frozen", acc1)


#fine tuning
basemodel.trainable=True
for layer in basemodel.layers[:-50]:
    layer.trainable=False


model.compile(
    optimizer=Adam(0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


history2=model.fit(
    xt,
    yt,
    epochs=5,
    validation_data=(xts,yts),
    verbose=2
)


loss2,acc2=model.evaluate(xts,yts)
print("finetuning acc",acc2)


plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])


plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])


plt.title("Tranfer learning")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend([
    "Frozen Train",
    "Frozen Validation",
    "FineTune Train",
    "FineTune Validation"
])
plt.show()
