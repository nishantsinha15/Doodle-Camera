from imageai.Detection import ObjectDetection
import os
import output
import cv2
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "model/yolo.h5"))
detector.loadModel()
print("Model Loaded")
import time
t = time.time()
detections = detector.detectObjectsFromImage(input_image="Data/dog_horse.jpg", output_image_path="Data/imagenew.png", minimum_percentage_probability=80)
print("Took ", time.time() - t)

objects = []
cord = []

for eachObject in detections:
    # print(list(eachObject.keys()))
    objects.append(eachObject["name"])
    cord.append(eachObject["box_points"])
    print(eachObject["name"] , " : " , eachObject["box_points"], eachObject['percentage_probability'] )

for i in range(len(cord)):
    temp1 = [cord[i][0], cord[i][1]]
    temp2 = [cord[i][2], cord[i][1]]
    temp3 = [cord[i][0], cord[i][3]]
    temp4 = [cord[i][2], cord[i][3]]
    cord[i] = [temp1, temp2, temp3, temp4]

print(objects)
print(cord)

out = output.final(cv2.imread("Data/dog_horse.jpg"), objects, cord)
cv2.imwrite('output', out)