import os
import cv2
from PIL import Image
import numpy as np
import pickle
#all code is used to train recognizer with various faces and save to trainner.yml
Base_dir=os.path.dirname(os.path.abspath(__file__))# current path
image_dir=os.path.join(Base_dir,"images")   #IMAGE save path

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()# face recognizer in cv2
current_id=0# give id to image of person
label_id={}
y_labels=[]
x_train=[]

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):#search file extension with jpg and png
            path=os.path.join(root, file)
            label=os.path.basename(root).replace(" ","-").lower()#displace image with dir name and replace space with - and save it at lower case
           # print(label,path)
            if not label in label_id:
                label_id[label]=current_id#give id to every unique person image
                current_id +=1#for same person id remain same
            id_=label_id[label]
            #print(label_id)

            pil_image=Image.open(path).convert("L")# convert in gray scale
            image_array=np.array(pil_image,"uint8")# type of image and convert pixel value to numpy array
           # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi=image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id_)# got label for every face

with open("label.pickle",'wb') as f:#save label in file pickle
    pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

