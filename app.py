# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:28:15 2021

@author: Sabyasachi
"""
from flask import Flask,request,jsonify,render_template
#import os
from flask_cors import CORS, cross_origin
#from covid_utils.utils import decodeImage
#from predict import covid

import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import base64
import io


app = Flask(__name__,template_folder='templates')
CORS(app)

model=model=load_model('modelCovid19_1.h5')

label_dict={'Covid19 Negative': 0, 'Covid19 Positive': 1}

def preprocess(img):
    #img = np.array(img,target=(100,100))  
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
    #img=np.array(img)/255.0
    img=np.array(img)
    if(img.ndim==3):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray=img
    
    gray=gray/255
    resized=cv2.resize(gray,(100,100))
    reshaped=resized.reshape(1,100,100,1)
    return reshaped
    print(img[0])
    print("Shape= ",img.shape)
    #img=np.reshape(img,(1,100,100,1))
    #return img

@app.route("/",methods=['GET'])
@cross_origin()
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        print('HERE')
        message = request.get_json(force=True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        dataBytesIO=io.BytesIO(decoded)
        dataBytesIO.seek(0)
        image = Image.open(dataBytesIO)
        test_image=preprocess(image)
        prediction=model.predict(test_image)
        accuracy=prediction[0][0]
        
        if accuracy<=0.5:
            #print("The person does not have symptoms of Covid")
            label="Covid19 Negative"
        else:
            #print("The person has symptoms of Covid")
            label="Covid19 Positive"
        print(prediction,label,accuracy)
        
        response = {'prediction': {'result': label}}
        print(response)
        return jsonify(response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)