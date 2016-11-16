#!/usr/bin/python3
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

def main():
    print("Reading", sys.argv[1])
    img = cv2.imread(sys.argv[1], 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply 3x3 square erosion 20 times
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones( (3, 3) ), iterations=20)
    
    # show img as found before finding contours
    plt.imshow(img)

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    plt.figure()
    plt.imshow(img)
    
    for contour in contours:
        xs = contour[:, 0, 0]
        ys = contour[:, 0, 1]
        #print("x", xs)
        #print("y", ys)
        
        plt.plot(xs, ys)
    plt.show()

if __name__ == "__main__":
    main()
