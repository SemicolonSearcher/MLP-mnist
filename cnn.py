import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense,Flatten,Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout

(xtrain,ytrain),(xtest,ytest)=mnist.load_data()

print("Training Image shape : ",xtrain.shape)
print("Training Labels shape: ",ytrain.shape)
print("Testing Image shape : ",xtest.shape) 
print("Testing Labels Shape : ",ytest.shape)

fig1=plt.figure(figsize=(10,5))

for i in range(10):

	plt.subplot(2,5,i+1)
	plt.imshow(xtrain[i],cmap='gray')
	plt.title(f"Digits : {ytrain[i]}")
	plt.axis('off')

plt.tight_layout()

xtrain=xtrain/255.0
xtest=xtest/255.0

xtraincnn=xtrain.reshape(-1,28,28,1)
xtestcnn=xtest.reshape(-1,28,28,1)

mlpmodel=Sequential([

	Input(shape=(28,28)),

	Flatten(),

	Dense(128,activation='relu'),

	Dense(128,activation='relu'),

	Dense(10,activation='softmax')

])

print("MLP MODEL SUMMARY")

mlpmodel.summary()

mlpmodel.compile(

	optimizer='adam',

	loss='sparse_categorical_crossentropy',

	metrics=['accuracy']

)

mlphistory=mlpmodel.fit(

	xtrain,
	ytrain,

	epochs=5,

	validation_data=(xtest,ytest)

)

mlploss,mlpaccuracy=mlpmodel.evaluate(xtest,ytest)

print("MLP MODEL LOSS",mlploss)

print("MLP TEST ACCURACY",mlpaccuracy)


cnnmodel=Sequential([

	Input(shape=(28,28,1)),

	Conv2D(32,(3,3),activation='relu'),

	MaxPooling2D((2,2)),

	Conv2D(64,(3,3),activation='relu'),

	MaxPooling2D((2,2)),

	Flatten(),

	Dense(128,activation='relu'),

	Dropout(0.5),

	Dense(10,activation='softmax')

])

print("CNN MODEL SUMMARY")

cnnmodel.summary()

cnnmodel.compile(

	optimizer='adam',

	loss='sparse_categorical_crossentropy',

	metrics=['accuracy']

)

cnnhistory=cnnmodel.fit(

	xtraincnn,
	ytrain,

	epochs=5,

	validation_data=(xtestcnn,ytest)

)

cnnloss,cnnaccuracy=cnnmodel.evaluate(xtestcnn,ytest)

print("CNN TEST ACCURACY",cnnaccuracy)

print("CNN TEST LOSS",cnnloss)


# SECOND CNN MODEL WITH DIFFERENT CONVOLUTION PARAMETERS

cnnmodel2=Sequential([

	Input(shape=(28,28,1)),

	Conv2D(16,(5,5),activation='relu'),

	MaxPooling2D((2,2)),

	Conv2D(32,(5,5),activation='relu'),

	MaxPooling2D((2,2)),

	Flatten(),

	Dense(64,activation='relu'),

	Dense(10,activation='softmax')

])

print("SECOND CNN MODEL SUMMARY")

cnnmodel2.summary()

cnnmodel2.compile(

	optimizer='adam',

	loss='sparse_categorical_crossentropy',

	metrics=['accuracy']

)

cnnhistory2=cnnmodel2.fit(

	xtraincnn,
	ytrain,

	epochs=5,

	validation_data=(xtestcnn,ytest)

)

cnnloss2,cnnaccuracy2=cnnmodel2.evaluate(xtestcnn,ytest)

print("SECOND CNN TEST ACCURACY",cnnaccuracy2)

print("SECOND CNN TEST LOSS",cnnloss2)


fig2=plt.figure(figsize=(10,5))

plt.plot(mlphistory.history['val_accuracy'])

plt.plot(cnnhistory.history['val_accuracy'])

plt.plot(cnnhistory2.history['val_accuracy'])

plt.title("MLP VS CNN")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['MLP','CNN 3x3','CNN 5x5'])

plt.grid(True)


featuremodel=Model(

	inputs=cnnmodel.inputs,

	outputs=cnnmodel.layers[0].output

)

sampleimage=xtestcnn[0].reshape(1,28,28,1)

featuremaps=featuremodel.predict(sampleimage)

fig3=plt.figure(figsize=(4,4))

plt.imshow(xtest[0],cmap='gray')

plt.title("Original image")

plt.axis('off')

fig4=plt.figure(figsize=(12,8))

for i in range(8):

	plt.subplot(2,4,i+1)

	plt.imshow(featuremaps[0, :, :, i],cmap='gray')

	plt.title(f"Feature map {i+1}")

	plt.axis('off')

plt.tight_layout()


# ACCURACY COMPARISON GRAPH

fig5=plt.figure(figsize=(8,5))

models=['CNN 3x3','CNN 5x5']

accuracies=[cnnaccuracy,cnnaccuracy2]

plt.bar(models,accuracies)

plt.title("Effect Of Convolution Parameters")

plt.ylabel("Accuracy")

plt.grid(True)


# PREDICTION USING CNN

prediction = cnnmodel.predict(xtestcnn)

predicted_digit = np.argmax(prediction[0])

print("\nPredicted Digit :", predicted_digit)

print("Actual Digit :", ytest[0])
	

# DISPLAY PREDICTED IMAGE

fig6 = plt.figure(figsize=(4,4))

plt.imshow(xtest[0], cmap='gray')

plt.title(f"Predicted : {predicted_digit}")

plt.axis('off')


plt.show()