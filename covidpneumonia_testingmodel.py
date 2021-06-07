import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


img_size=50
def covidtest(fln):

 filepath = 'YOUR PATH TO YOUR SAVED MODEL'
 
 model=load_model(filepath,compile=True)
 


 print("model loaded :)")

 
 img = cv2.imread(fln)
 img = cv2.resize(img,(img_size,img_size))
 img = np.reshape(img,[-1,img_size,img_size,3])
 img = np.array(img)
 classes = model.predict_classes(img)
 return(classes)
 


