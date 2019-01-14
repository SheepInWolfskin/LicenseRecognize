import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('ez.png')
# use gaussina blur to denoise
blur = cv2.blur(image,(5,5))

# convert to gray image
img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
print(img.shape)

# detect edge
edges = cv2.Canny(img,80,350)

# OTSU
ret, img_thre = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)



# cv2.imshow('new', img_thre)
# cv2.waitKey(0)

# record number of white pixel of every column
white = []
# record number of black pixel of every column
black = []

height = img_thre.shape[0]
width = img_thre.shape[1]
white_max = 0
black_max = 0

# calculate sum of white and black pixels in each column
for i in range(width):

    # sum of white pixels in this column
    temp_white = 0

    # sum of black pixels in this column
    temp_black = 0

    for j in range(height):
        if img_thre[j][i] == 255:
            temp_white += 1
        if img_thre[j][i] == 0:
            temp_black += 1

    white_max = max(white_max, temp_white)
    black_max = max(black_max, temp_black)

    white.append(temp_white)
    black.append(temp_black)


check_back = False
if black_max > white_max:
    check_back = True


def find_end(start_):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        rate1 = 0.93
        b_or_w = (black[m] if check_back else white[m])
        common = (rate1 * black_max if check_back else rate1 * white_max)
        if b_or_w > common:
            end_ = m
            break
    return end_


n = 1
start = 1
end = 2
while n < width - 2:
    n += 1
    rate = 0.05
    check_bg = (white[n] if check_back else black[n])
    bg = (rate * white_max if check_back else rate * black_max)
    # check the background is white or black
    if check_bg > bg:
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            opt = img_thre[1:height, start:end]
            cv2.imwrite('new{0}.jpg'.format(n), opt)
            cv2.imshow('new', opt)
            cv2.waitKey(0)


# cv2.imshow('new', threshold)
# cv2.waitKey(0)


# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
