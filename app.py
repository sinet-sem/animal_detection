import tensorflow as tf
import keras
from flask import Flask, render_template, request, jsonify
from PIL import Image
import os, io, base64, sys
#from keras.preprocessing.image import img_to_array
import json, numpy as np
import cv2
import tensorflow

app = Flask(__name__)
# my_model = keras.saving.load_model("my_model.h5")


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        response = {"success": False}

        if 'file' in request.files:
            # 1. read & pre-proces image that we pick
            image = request.files["file"].read()
            image = Image.open(io.BytesIO(image))
            
            if image.mode != "RGB":
                image = image.convert("RGB")

            image = image.resize((50,50)) 
            image = tf.keras.utils.img_to_array(image) # shape (32,32,3)
            image = np.expand_dims(image, axis=0) # shape (1,32,32,3)

            # 2. load the trained model & predict on the above image
            model = keras.saving.load_model("my_model.h5")
            y_pred = model.predict(image,verbose=2)

            
            # filter the label code that has the highest probability
            label_code = np.argmax(y_pred)
            class_label = ['Bear', 'Brown bear', 'Butterfly', 'Camel', 'Caterpillar', 
                           'cats', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'dogs', 'Duck', 
                           'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe', 'Goldfish', 'Goose', 
                           'Hedgehog', 'Horse', 'Lion', 'Lizard', 'Monkey', 'Mouse', 'Panda', 'Pig', 
                           'Rabbit', 'Seahorse', 'Shark', 'Snake', 'Squid', 'Tiger', 'Zebra']
            # class_label = ['cats', 'dogs', 'panda']
            response['label'] = class_label[label_code]
            response['class_proba'] = json.loads(json.dumps(y_pred[0].tolist()))

            response['class_label'] = class_label
            response['success'] = True

            return jsonify(response)
    
    return render_template('index.html')

# main function
if __name__ == '__main__':
   app.run(debug=True)



# my_model = keras.saving.load_model("my_model.h5")


# def convert_to_array(img):
#     im = cv2.imread(img)
#     img = Image.fromarray(img, 'RGB')
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
# convert_to_array("f288c2385b3935c0.jpg")
# predict_animal("f288c2385b3935c0.jpg")
# predict_animal("c:/UP/project/dataSet/train/Bear/f288c2385b3935c0.jpg")
# predict_animal("c:/Users/Admin/Downloads/train/Pig/d7c11e86dd8ffd38.jpg")

# from keras.models import load_model
# import cv2
# import numpy as np



# classes = np.argmax(my_model.predict(img), axis = -1)

# print(classes)

# names = [class_names[i] for i in classes]

# print(names)