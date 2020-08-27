

import sys, os
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

df=pd.read_csv('gdrive/My Drive/fer2013.csv')

# print(df.info())
# print(df["Usage"].value_counts())

# print(df.head())
x_train,y_train,x_test,y_test=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           x_train.append(np.array(val,'float32'))
           y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           x_test.append(np.array(val,'float32'))
           y_test.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")


features = 64
labels = 7
batch_size = 128
epochs = 400
width, height = 48, 48


x_train = np.array(x_train,'float32')
y_train = np.array(y_train,'float32')
x_test = np.array(x_test,'float32')
y_test = np.array(y_test,'float32')

y_train=np_utils.to_categorical(y_train, num_classes=labels)
y_test=np_utils.to_categorical(y_test, num_classes=labels)

x_train -= np.mean(x_train, axis=0)
x_train /= np.std(x_train, axis=0)

x_test -= np.mean(x_test, axis=0)
x_test /= np.std(x_test, axis=0)

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)

x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

#1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(x_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.45))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.45))


model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.45))


model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.45))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(labels, activation='softmax'))

# model.summary()

datagen = ImageDataGenerator(featurewise_center=True, 
                             featurewise_std_normalization=True,
                             zoom_range = 0.15,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range = 0.15,
                             horizontal_flip=True,
                             fill_mode = "nearest")



#Compliling the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr = 0.0003),
              metrics=['accuracy'])


datagen.fit(x_train)

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', restore_best_weights = True)

#Training the model
#model.fit(x_train, y_train,
         #batch_size=batch_size,
         #epochs=epochs,
         #verbose=1,
         #validation_data=(x_test, y_test),
         #shuffle=True)
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size= batch_size), validation_data = (x_test, y_test), steps_per_epoch=len(x_train)/batch_size, epochs=epochs, verbose =1)

#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
