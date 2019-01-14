import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def stretch(img):
    max = float(img.max())
    min = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (max - min)) * img[i, j] - (255 * min) / (max - min)

    return img


def find_rectangle(input_array):
    y, x = [], []

    for p in input_array:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]


def locate_license(img, orgimg):

    # find rectangles in pictures
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find three biggest rectangles
    blocks = []
    for c in contours:
        # find the left up and right down, calculate the size / length
        r = find_rectangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = (r[2] - r[0]) / (r[3] - r[1])
        blocks.append([r, a, s])

    # select largest 3 areas
    blocks = sorted(blocks, key=lambda b: b[2])[-3:]

    # use color to find the area which is most similar to license
    max_w = 0
    max_i = -1
    for i in range(len(blocks)):
        b = orgimg[blocks[i][0][1]:blocks[i][0][3], blocks[i][0][0]:blocks[i][0][2]]

        # RGB to HSV
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

        # the area of blue
        lower = np.array([20, 50, 50])
        upper = np.array([140, 255, 255])

        # cover the area with threshold
        mask = cv2.inRange(hsv, lower, upper)

        # calculate the weight
        w1 = 0
        for m in mask:
            w1 += m / 255

        w2 = 0
        for w in w1:
            w2 += w

        # find the area with largest weight
        if w2 > max_w:
            max_i = i
            max_w = w2

    return blocks[max_i][0]


def find_license(img):
    img = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray stretch
    stretch_img = stretch(gray_img)

    # use open operation to denoise
    r = 16
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(kernel, (r, r), r, 1, -1)

    openingimg = cv2.morphologyEx(stretch_img, cv2.MORPH_OPEN, kernel)
    strtimg = cv2.absdiff(stretch_img, openingimg)

    # binary the image
    ret, binaryimg = cv2.threshold(strtimg, 100, 255, cv2.THRESH_BINARY)

    # detect the edges with canny edge detect
    canny = cv2.Canny(binaryimg, binaryimg.shape[0], binaryimg.shape[1])

    # use close operation
    kernel = np.ones((5, 19), np.uint8)
    closingimg = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    # use open operation
    openingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel)

    # use open operation again
    kernel = np.ones((11, 5), np.uint8)
    openingimg = cv2.morphologyEx(openingimg, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('img',openingimg)
    # cv2.waitKey(0)

    # get rid of small area, locate the license
    rect = locate_license(openingimg, img)

    return rect, img


if __name__ == '__main__':
    # orgimg = cv2.imread('test.png')
    # orgimg = cv2.imread('test3.jpeg')

    orgimg = cv2.imread('easy.jpeg')
    rect, img = find_license(orgimg)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    cv2.imshow('img', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
