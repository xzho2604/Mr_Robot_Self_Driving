import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers.core import Dense, Activation
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.optimizers import Adam

import numpy as np
import cv2

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model


# ================================================================================
# standard CNN model
def build_model():
    '''
    Build CNN model for light color classification
    '''
    num_classes = 3
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding='same',
                     input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    plot_model(model, to_file='model.png')

    return model

# --------------------------------------------------------------------------------
# transfer leaning using mobiile net as base model
def build_mobile_model():
    # use the mobile net base model fix base model params for training
    base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    # add some layers for training on top of base model
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x)
    x=Dense(512,activation='relu')(x) #dense layer 2
    preds=Dense(3,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)
    # check the model architect
    for i,layer in enumerate(model.layers):
      print(i,layer.name)

    # or if we want to set the first 50 layers of the network to be non-trainable
    for layer in model.layers[:20]:
        layer.trainable=False
    for layer in model.layers[20:]:
        layer.trainable=True

    return model
# ================================================================================
model = build_model()
# generate training data automatically given training dir with subdir name as class name
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
train_generator=train_datagen.flow_from_directory('training_images',
                                                 target_size=(32,32),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
print(train_generator.class_indices) # print class and indexes

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)

model.save("my_model.h5")


