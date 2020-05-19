import cv2
import numpy as np
import math

y_upperlimit = 195  # vga
y_lowerlimit = 270  # vga
min_value = 60  # Canny
max_value = 100  # Canny
hough_thresh = 105


def nothing(x):
    pass


# load image
def getImage(image_path_left, image_path_right, path_index, get_stereo=False):
    if get_stereo:
        img_left = cv2.imread(image_path_left[path_index])
        img_right = cv2.imread(image_path_right[path_index])
        img = cv2.hconcat((img_left, img_right))
    else:
        img = cv2.imread(image_path_left[path_index])
    return img


# get vertical lines
def getVertical():
    global img, min_value, max_value, mask
    edges = cv2.Canny(img.copy(), min_value, max_value)
    vertical = edges.copy()
    vertical_size = int(rows / 30)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical = np.multiply(vertical, mask)
    return vertical


# get horizontal lines
def getHorizontal(bins):
    global img, min_value, max_value, mask
    edges = cv2.Canny(img.copy(), min_value, max_value)
    horizontal = edges.copy()
    horizontal_size = int(cols / max(1, bins))
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    horizontal = np.multiply(horizontal, mask)
    return horizontal


def houghLines(edges, hough_thresh, theta_range):
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdst = cdst.copy()
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i, 0, 0]
            theta = lines[i, 0, 1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            if (abs(np.pi / 2 - theta) > theta_range * 0.1 / 180 * np.pi):
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
            elif (abs(np.pi / 2 - theta) < theta_range * 0.025 / 180 * np.pi):
                cv2.line(cdst, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('hough lines', cdst)


path_index = 0
image_path_left = ['/Users/luchixiang/Downloads/test1.jpg',
                   '/Users/luchixiang/Downloads/test1.jpg',
                   '/Users/luchixiang/Downloads/test1.jpg',
                   '/Users/luchixiang/Downloads/test1.jpg'
                   ]

image_path_right = [
    '/Users/luchixiang/Downloads/test1.jpg',
    '/Users/luchixiang/Downloads/test1.jpg',
    '/Users/luchixiang/Downloads/test1.jpg'
]

img = getImage(image_path_left, image_path_right, path_index)
print(img.shape)
cv2.imshow('original', img)

rows, cols, channels = np.shape(img)
mask_upper = np.zeros((y_upperlimit, cols), np.uint8)
mask_middle = np.ones((y_lowerlimit - y_upperlimit, cols), np.uint8)
mask_lower = np.zeros((rows - y_lowerlimit, cols), np.uint8)
mask = np.concatenate((mask_upper, mask_middle, mask_lower), axis=0)

cv2.namedWindow('Canny')
cv2.createTrackbar('min', 'Canny', min_value, 256, nothing)
cv2.createTrackbar('max', 'Canny', max_value, 256, nothing)
cv2.createTrackbar('hough_thresh', 'Canny', hough_thresh, 200, nothing)
cv2.createTrackbar('theta_range', 'Canny', 20, 200, nothing)

while (1):
    min_value = cv2.getTrackbarPos('min', 'Canny')
    max_value = cv2.getTrackbarPos('max', 'Canny')
    hough_thresh = cv2.getTrackbarPos('hough_thresh', 'Canny')
    theta_range = cv2.getTrackbarPos('theta_range', 'Canny')
    edges = cv2.Canny(img.copy(), min_value, max_value)
    edges = np.multiply(edges, mask)
    # eraze vertical lines
    edges = cv2.bitwise_xor(edges, getVertical())
    # eraze horizontal lines
    # edges = cv2.bitwise_xor(edges, getHorizontal(horiz_bins))
    cv2.imshow('edges', edges)
    houghLines(edges, hough_thresh, max(theta_range, 1))
    key = cv2.waitKey(1)
    if key == ord("n"):
        path_index = min(path_index + 1, 22)
        img = getImage(image_path_left, image_path_right, path_index)
        cv2.imshow('original', img)
    elif key == ord("b"):
        path_index = max(path_index - 1, 0)
        img = getImage(image_path_left, image_path_right, path_index)
        cv2.imshow('original', img)
    elif key == ord("e"):
        break

cv2.destroyAllWindows()
