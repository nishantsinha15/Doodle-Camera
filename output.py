import cv2
import numpy as np
import ndjson
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA


# a = np.load('full_numpy_bitmap_flower.npy')
# a = np.load('full_numpy_bitmap_flower.npy')
# img = cv2.imread('flower.jpg')

def canny(img):
    # img = cv2.imread('flower.jpg')
    # img = cv2.blur(img,(3,3))
    img = cv2.Canny(img, 100, 200)
    cv2.imwrite('After canny.png', img)
    return img


def sketch(img):
    img = canny(img)
    temp = int((np.amin(np.array([img.shape[0], img.shape[1]]))) / 60)
    kernel = np.ones((temp, temp), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.resize(img_dilation, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite("After sketch.png",img_dilation )
    return img_dilation


# img = cv2.imread('fan.jpeg',0)
# # img = canny(img)
# img = sketch(img)
# plt.imshow(img)
# cv2.imwrite('fan_sketch_28.jpg',img)
# plt.show()

def map(img_f, img, size_x, size_y, x, y):
    img = cv2.resize(img, (size_y, size_x), interpolation=cv2.INTER_AREA)
    img_f[x:x + size_x, y:y + size_y] = img_f[x:x + size_x, y:y + size_y] + img
    return img_f


# img = cv2.imread('dog_sketch_256.jpg',0)
# img1 = cv2.imread('flower_sketch_256.jpg',0)
# img_f = np.ones((400,400))
# img_f = map(img_f,img,200,300,100,100)
# img_f = map(img_f,img1,200,300,50,100)
# plt.imshow(img_f)
# cv2.imwrite('mapping_try2.jpg',img_f)
# plt.show()

def cosine_sim(a, b):
    c = np.dot(a, b)
    c /= np.linalg.norm(a) * np.linalg.norm(b)
    return c


def preprocess(img):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.filter2D(img, -1, np.ones((10, 10), np.float32) / 100)
    kernel = np.ones((6, 6), np.uint8)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("Preproceed doodle.png", img)
    return img


def crop(img, coordinate):
    crp = img[coordinate[0][0]:coordinate[2][0], coordinate[0][1]:coordinate[1][1]]
    return crp


def set(typee):
    # return set of doodle list of 28x28
    dir = os.listdir('Data/')
    for file in dir:
        if ('full_numpy_bitmap_' + typee + '.npy' == file):
            return np.load('data/' + file)


# print(set('car').shape)
def closest(sketch, doodle_set):
    pca = PCA(0.95)
    pca.fit(doodle_set)
    newdata = pca.transform(doodle_set)
    sketch = pca.transform([sketch])
    closeness = []
    N = 0
    te = 1000
    ii = 0
    for i, n in zip(newdata, range(len(newdata))):
        t = abs(cosine_sim(sketch, i))
        closeness.append(t)

    neww = [x for _, x in sorted(zip(closeness, doodle_set))]
    return ([neww[999], neww[998], neww[997], neww[996], neww[995]])


# return [doodle_set[999]]
def final(img, typee, coordinate):
    fin = np.zeros(img.shape)
    for i, j in zip(typee, coordinate):
        crp = crop(img, j)
        doodle = sketch(crp).flatten()
        doodle_set = set(i)[0:1000]

        closest_doodle = closest(doodle, doodle_set)[0]
        closest_doodle = np.asarray(closest_doodle)

        closest_doodle1 = closest_doodle.astype(float).reshape((28, 28))
        resized_doodle = preprocess(closest_doodle1)
        fin = map(fin, resized_doodle, j[2][0] - j[0][0], j[1][1] - j[0][1], j[0][0], j[0][1])
    return fin


# img = cv2.imread('fan.jpeg', 0)
# fin = final(img, ['fan'], [[[0, 0], [0, 600], [400, 0], [400, 600]]])
# plt.imshow(fin)
# plt.show()

typee = ['horse', 'dog']
cord = [[[15, 94], [527, 94], [15, 377], [527, 377]], [[554, 255], [758, 255], [554, 377], [758, 377]]]
img_loc = 'Data/dog_horse.jpg'
img = cv2.imread(img_loc, 0)
fin = final(img, typee, cord)
