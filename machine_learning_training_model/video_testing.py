#model training credits goes to https://github.com/GantMan/nsfw_model
#you can downloads weight files from above mentioned url

#use :  python -u 'video url'
#python video_testing.py -u 'https://instagram.flko4-1.fna.fbcdn.net/v/t50.2886-16/108328813_155624022764916_4951514170493052098_n.mp4?_nc_ht=instagram.flko4-1.fna.fbcdn.net&_nc_cat=100&_nc_ohc=YieGaUgyFfMAX__O8m6&oe=5F28F491&oh=ab1aa81d1e6c6cedca3fc2a9487445a9'
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model,model_from_json
import tensorflow_hub as hub
from threading import Thread
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u','--url',required = True,
                    help = "give the video url file")

args = vars(parser.parse_args())

model = load_model('./',custom_objects={'KerasLayer':hub.KerasLayer})  #give the saved_model path
model.summary()
#model.load_weights('saved_model_weights.h5')

# for layer in model.layers:
#     print(layer.weight())
cap = cv2.VideoCapture(args['url'])
while(cap.isOpened()):
    _, frame = cap.read()
    frame_1 = cv2.resize(frame,(224,224),interpolation = cv2.INTER_NEAREST)
    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
    #print(frame.shape)
    frame_1 = np.expand_dims(frame_1,axis = 0)
    prediction = model.predict(frame_1)
    #print(prediction)
    val = np.argmax(prediction)
    categories = ['drawings', 'hentai', 'neutral','porn','sexy']

    cat = categories[val]
    if(cat != "porn"):
        frame = cv2.putText(frame,str(cat), (90,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    else:
        frame = cv2.blur(frame,(140,140)) 
        frame = cv2.putText(frame,str(cat), (90,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow('output_frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break