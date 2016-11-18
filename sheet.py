#!/usr/bin/python3
import cv2
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle

def processQr(contour):
    print("Processing Qr code")

def moment_diff(mom1, mom2, method=2):
    sum = 0
    if method == 2:
        selection_func = np.argmax
    else:
        selection_func = np.argmin
    for a, b in zip(mom1, mom2):
        if method == 1:
            sum += abs(a + b)

        if method == 2:
            sum += abs(1/a + 1/b)

        if method == 3:
            sum += abs((a + b) / a)
    return (sum, selection_func)

def main():
    a4 = 0
    with open('moments.txt', "rb") as f:
        a4 = pickle.load(f)
    
    print("a4 hu:", a4)

    HuThreshold = 15 # Hu Min Threshold (method 3)

    print("Reading", sys.argv[1])
    img = cv2.imread(sys.argv[1], 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    original = img
    img = cv2.GaussianBlur(img, (15, 15), 0)

    blurred = img
    
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    b = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
    
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones( (3, 3) ), iterations = 2)
    b = cv2.morphologyEx(b, cv2.MORPH_ERODE, np.ones( (3, 3) ), iterations = 2)
    cv2.subtract(img, b, img)

    # show img as found before finding contours
    plt.imshow(img)

    mom = cv2.moments(img, binaryImage=True)

    mom = cv2.HuMoments(mom).flatten()
    #mom = -np.sign(mom) * np.log(mom)
    print("contour moment:", mom)
    
    # uncomment to save moments
    #with open("moments.txt", "wb") as f:
    #    pickle.dump(mom, f)

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    plt.figure()
    plt.imshow(original, cmap=plt.cm.gray)

    moments = []

    for contour in contours:
        xs = contour[:, 0, 0]
        ys = contour[:, 0, 1]

        mom2 = cv2.moments(contour, binaryImage=True)
        area = mom2['m00']
        coverage = area / (img.shape[0] * img.shape[1])
        
        mom2 = cv2.HuMoments(mom2).flatten()
        #mom2 = -np.sign(mom2) * np.log(mom2)
        
        diff = abs(mom2 - a4)

        if coverage > 0.1:
            print("coverage", coverage)
            print("moment", mom2)
            print("diff", diff)
            print("----------")
            match = moment_diff(a4, mom2, method=3)
            print(match)
            print("----------")
            
            moments += (match, contour)

    print("moments:", [a for a,b in moments[0::2]])  

    hu_val = moments[0][1]([a for a,b in moments[0::2]])

    if moments[0::2][hu_val][0] < HuThreshold:
        contour = moments[1::2][hu_val]
        
        xs = contour[:, 0, 0]
        ys = contour[:, 0, 1]

        # process QR
        processQr(contour)
        
        plt.plot(xs, ys, linewidth=5, color='cyan')
    plt.show()

if __name__ == "__main__":
    main()
