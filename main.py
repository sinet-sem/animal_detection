from PIL import Image
import numpy as np
import os
import keras
from keras import layers
import cv2
data=[]
labels=[]
TF_ENABLE_ONEDNN_OPTS=0
# Location of main dataset
# base_dir  = 'c:/UP/project/dataSet/'
base_dir  = 'c:/UP/project/dataSet/'

train_dir = os.path.join(base_dir, 'train')
# test_dir = os.path.join(base_dir, 'test')
classes = os.listdir(train_dir)
num_classes = len(classes)
# print(f'Number of class : {num_classes}')

numberOfLabel = 0
for class_name in classes:
    className=os.listdir(train_dir+"/"+class_name)
    for img in className:
        if img != "Label":
            imag=cv2.imread(train_dir+"/"+class_name + "/" +img)
            img_from_ar = Image.fromarray(imag, 'RGB')
            resized_image = img_from_ar.resize((50, 50))
            data.append(np.array(resized_image))
            # data.append(img)
            labels.append(numberOfLabel)
    numberOfLabel +=1
animals=np.array(data)
labels=np.array(labels)

np.save("animals",animals)
np.save("labels",labels)

animals=np.load("animals.npy")
labels=np.load("labels.npy")

s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

# num_classes=len(np.unique(labels))
# print(f'num_class 1: {num_classes}')
data_length=len(animals)

(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)


# #make model
model=keras.Sequential()
model.add(keras.layers.Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
# model.add(keras.Input(shape=(50,50,3)))
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



model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=32
          ,epochs=50,verbose=2)


print("loop finish____________")
score = model.evaluate(x_test, y_test, verbose=2)
print('\n', 'Test accuracy:', score[1])

model.save('my_model_test1.h5', save_format='h5')


#******************** test predit image *****************************
# my_model = keras.saving.load_model("my_model.h5")


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
# # predict_animal("c:/UP/project/dataSet/train/Bear/f288c2385b3935c0.jpg")
# predict_animal("c:/Users/Admin/Downloads/train/Pig/d2f1628b99195c2c.jpg")

# dataSet/train/Lion/0c024331927586b8.jpg
# dataSet/test/Elephant/1a676e236a83f4ec.jpg
# dataSet/test/Horse/3a80fd5cc3e2eec4.jpg
# c:\Users\Admin\Downloads\train\Bear\4fcee0e067c2cd53.jpg
# dataSet/train/Duck/0a5614dc0eb7a332.jpg
# dataSet/train/Bear/f288c2385b3935c0.jpg
# y_pred = my_model.predict(x_test,verbose=2)
# print(f'predit data : {y_pred[1]}')
# digit_index = 5
# img = x_test[digit_index].reshape((50,50,3))
# print(f'image testing: {img}')
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# plt.imshow(img)
# plt.show()

# import numpy as np
# digit = np.argmax(y_pred[digit_index])
# score=my_model.predict(img,verbose=2)
# print(f'score: {score}')
# print("The model predict above digit image to number: " + str(digit))
