import numpy as np
import cv2
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = os.getcwd()
data_path = os.path.join(path,'images')
IDS = []
Faces =[]

ddd = {}
count=1
for files,roots,dirs in os.walk(data_path):
    for root in roots:
        ddd[''+root+'']=count
        count+=1
    #for root in roots:
    #    IDS.append(dummy[''+root+''])
    for di in dirs:
        imagep = os.path.join(files,di)
        if imagep.endswith('.jpg'):
            ids = files.split("\\")[2]
            IDS.append(ddd[''+ids+''])
            image = cv2.imread(imagep)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(image,(250,250))
            Npface = np.array(resized,'uint8')
            Faces.append(Npface)
print(ddd)
IDS = np.array(IDS)
recognizer.train(Faces,IDS)
os.chdir(path)
recognizer.save('trained_data.yml')
