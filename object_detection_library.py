from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "model/yolo.h5"))
detector.loadModel()

import time
t = time.time()
detections = detector.detectObjectsFromImage(input_image="image2.jpg", output_image_path="imagenew.png")
print("Took ", time.time() - t)

for eachObject in detections:
    # print(list(eachObject.keys()))
    print(eachObject["name"] , " : " , eachObject["box_points"] )