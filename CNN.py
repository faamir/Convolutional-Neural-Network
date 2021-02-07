from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_metrics
import matplotlib.pyplot as plt
import time

# split data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#plot a simple digit number
img_idx = 1000
print("Example of a digit: ", train_y[img_idx]) # The label is 0
#plt.figure(1)
#plt.imshow(train_x[img_idx], cmap='Greys')
#plt.title(' Example of a digit ')
#plt.show()

#reshape data to 2D [X,28*28]
train_x = train_x.reshape(train_x.shape[0], 784)
test_x = test_x.reshape(test_x.shape[0], 784)
input_shape = (784,)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

#Normalize data
train_x = train_x/255
test_x = test_x/255
print('train_x shape:', train_x.shape)
nclass = 10

# change class to binary class matrix
train_y = keras.utils.to_categorical(train_y, nclass)
test_y = keras.utils.to_categorical(test_y, nclass)

#timing
start = time.time()

#MLP Model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=input_shape))
model.add(Dropout(0.35))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(nclass, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', 
              metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])

hist=model.fit(train_x, train_y, batch_size=64, epochs=20, 
               verbose=2, validation_split=0.15)
accuray = model.evaluate(test_x, test_y, verbose=0)
print("Test Accuracy: ", (accuray[1]*100))
print('Test loss:', accuray[0])
end = time.time()
print("Time elapsed (seconds): ", end - start)

#Plot accuracy for train and validation
plt.figure(2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'], '--')
plt.title('Accuracy of the model')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

#Plot loss for train and validation
plt.figure(3)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'],'--')
plt.title('Loss of the model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# plot Precision & Recall
plt.figure(4)
plt.plot(hist.history['precision'])
plt.plot(hist.history['recall'], '--')
plt.title(' Precision & Recall ')
plt.ylabel('%')
plt.xlabel('epoch')
plt.legend(['Precision', 'Recall'], loc='lower right')
plt.show()

img_idx = 1005
plt.figure(5)
plt.imshow(test_x[img_idx].reshape(28, 28), cmap='Greys')
plt.title(' Example of predict a digit ')
plt.show()
pred = model.predict(test_x[img_idx].reshape(1, 784))
print("predicted output: ",pred.argmax())
