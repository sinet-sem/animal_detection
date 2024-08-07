# A libraries to avoid keras and tensorflow warnings
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# A libraries to avoid other warnings
from warnings import filterwarnings
filterwarnings('ignore')

import os                                # To work with operation system comands
import pandas as pd                      # To work with DataFrames
import numpy as np                       # To work with arrays
import random                            # To generate random number and choices
import matplotlib.pyplot as plt          # To create plots and visualizations
import seaborn as sns                    # To create plots and visualizations
from termcolor import colored            # To create colorfull output
from PIL import Image                    # To read images from source

import tensorflow as tf                  # Main Franework
import keras                             # To create and manage deep neural networks
from keras import layers
import cv2
sns.set_style('darkgrid')

#### Location of main dataset
base_dir  = 'c:/UP/project/dataSet/'

##### Show main directory containers
os.listdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
# train_dir = os.path.join(base_dir, 'data_train')

classes = os.listdir(train_dir)
num_classes = len(classes)
print(f'classes : {classes}')
print(f'Number of class : {num_classes}')

# Create a DataSet
BATCH_SIZE = 32
IMAGE_SIZE = (50, 50)

train_full = keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels='inferred',
    label_mode='categorical',
    class_names=classes,
    seed=42,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,    
)
# data_test = keras.utils.image_dataset_from_directory(
#     directory=test_dir,
#     labels='inferred',
#     label_mode='categorical',
#     class_names=classes,
#     seed=42,
#     shuffle=True,
#     batch_size=BATCH_SIZE,
#     image_size=IMAGE_SIZE,    
# )

train_full = train_full.shuffle(1024).prefetch(tf.data.AUTOTUNE)



num_all_batches = len(list(train_full))
print(f'Number of all Batches : {num_all_batches}')

num_train_batches = int(num_all_batches * 0.9)
num_valid_test_batches = int(num_all_batches - num_train_batches)

print(' Target : ')
print('-'*35)
print(f'Number of  Train  batches : {num_train_batches}')
print(f'Number of Validation batches : {num_valid_test_batches}')


train_ds = train_full.take(num_train_batches)

remain = train_full.skip(num_train_batches)

valid_ds = remain.take(num_valid_test_batches//2)
test_ds = remain.skip(num_valid_test_batches//2)



model=keras.Sequential()
# model.add(keras.layers.Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50, 50,3)))
model.add(keras.Input(shape=(50,50,3)))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500,activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(num_classes,activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    train_ds,                                          # Dataset to train model
    epochs=50,                                        # Number of epochs to train
    validation_data=valid_ds,                          # Validation dataset
    verbose=2
)

print("loop finish____________")
score = model.evaluate(test_ds, verbose=2)
print('\n', 'T est accuracy:', score[1])

model.save('my_model_test.h5', save_format='h5')

# model.save('my_model.keras', save_format='keras')



# ********** reload model to test ****************
# my_model = keras.saving.load_model("my_model.h5")
# digit_index = 100
# img = valid_ds[digit_index].reshape((50,50,3))

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# plt.imshow(img)
# plt.show()

# def convert_to_array(img):
#     im = cv2.imread(img)
#     img = Image.fromarray(im, 'RGB')
#     image = img.resize((50, 50))
#     return np.array(image)
# def get_animal_name(label):
#     if label==0:
#         return "Bear"
#     if label==1:
#         return "Duck"
#     if label==2:
#         return "Elephant"
#     if label==3:
#         return "Fish"
#     if label==4:
#         return "Frog"
#     if label==5:
#         return "Horse"
#     if label==6:
#         return "Lion"
#     if label==7:
#         return "Pig"
# def predict_animal(file):
#     print("Predicting .................................")
#     ar=convert_to_array(file)
#     ar=ar/255
#     label=1
#     a=[]
#     a.append(ar)
#     a=np.array(a)
#     score=my_model.predict(a,verbose=2)
#     print(f'score : {score}')
#     label_index=np.argmax(score)
#     print(label_index)
#     acc=np.max(score)
#     animal=get_animal_name(label_index)
#     print(animal)
#     print("The predicted Animal is a "+animal+" with accuracy = "+str(acc))
# predict_animal("c:/UP/project/dataSet/train/Bear/f288c2385b3935c0.jpg")