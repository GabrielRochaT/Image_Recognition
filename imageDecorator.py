import cv2
import os
from pathlib import Path

folder = "imgs"
num_images = len(os.listdir(folder))
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
files = os.listdir(folder)

classFile = "assets/coco.names"
classNames = []

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "assets/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "assets/frozen_inference_graph.pb"

#CONFIGURACAO E PARAMETRIZACAO DA REDE
net = cv2.dnn.DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, conf, nms, draw=True, objects=[]):
    classIDs, confs, bbox, = net.detect(img, confThreshold=conf, nmsThreshold=nms,)
    count = 0;
    if len(classIDs)!=0:
        for classId, confidence, box in zip(classIDs.flatten(), confs.flatten(), bbox): 
            count +=1;
            className= classNames[classId-1]
            if draw:
                cv2.rectangle(img, box, color=(0,255,0), thickness=2)
                cv2.putText(img, className.upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return 0

for file in files:
    if Path(file).suffix.lower() in valid_extensions:
        image_path = os.path.join(folder, file)
        image = cv2.imread(image_path)
        result = getObjects(image, 0.50, 0.3, objects=[])
        if image is None:
            continue
        else:
            cv2.namedWindow("Imagens", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Imagens", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Imagens", image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()