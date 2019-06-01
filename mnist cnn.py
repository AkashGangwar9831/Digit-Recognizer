# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 10:01:24 2019

@author: Akash
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


train_data = "F:/digit-recognizer/train.csv"
test_data = "F:/digit-recognizer/test.csv"
output_file = "F:/digit-recognizer/CNN.csv"

raw_data = np.loadtxt(train_data, skiprows=1, dtype='int', delimiter=',')

X=raw_data[:,1:]
y=raw_data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
#1
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',input_shape = (28, 28, 1)))
model.add(BatchNormalization())
#2
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
#3
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))
#4
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
#5
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
#6
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
#1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
#2
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
#output layer
model.add(Dense(10, activation='softmax'))


datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)




hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=50, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(X_test[:500,:], y_test[:500,:]), #For speed
                           callbacks=[annealer])


final_loss, final_acc = model.evaluate(X_test, y_test, verbose=0)
print("Final loss: .4f  final accuracy: .4f".format(final_loss, final_acc))

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()

y_hat = model.predict(X_test)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)

#validation
mnist_testset = np.loadtxt(test_data, skiprows=1, dtype='int', delimiter=',')
x_val = mnist_testset.astype("float32")
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)/255
y_hat = model.predict(x_val, batch_size=64)
y_pred = np.argmax(y_hat,axis=1)

#final result to csv
with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(y_pred)) :
        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))

