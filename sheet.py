#!/usr/bin/python3
import cv2
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle
import skimage.io

def processQr(contour):
    print("Processing Qr code")

def train(itrain=False):
    print("Training moments")
    n = 0
    for e in sys.argv[3::1]:
        sum = 0
        n += 1
        if (itrain):
            print("Reading", e)
            e = cv2.imread(e, 0)
            mom = cv2.moments(e)
            mom = cv2.HuMoments(mom)
        else:
            with open(e, "rb") as f:
                mom = pickle.load(f)
                print("Loaded", mom)
        sum += mom
    print("New moment:", sum)
    with open(sys.argv[2], "wb") as f:
        pickle.dump(sum / n, f)

def hu_detect(moment, img):
    HuThreshold = 0.05 # Hu Min Threshold (method 3)

    original = img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

    moments = []

    for contour in contours:
        contour_moment = cv2.moments(contour, binaryImage=True)
        area = contour_moment['m00']
        coverage = area / (img.shape[0] * img.shape[1])
        
        contour_moment = cv2.HuMoments(contour_moment).flatten()
        #contour_moment = -np.sign(contour_moment) * np.log(contour_moment)
        
        diff = abs(contour_moment - moment)

        if coverage > 0.1:
            match = cv2.matchShapes(np.ones((297, 210)), contour, 3, 0.0)
            print("coverage", coverage)
            print("moment", contour_moment)
            print("diff", diff)
            print("----------")
            print("match:", match)
            print("----------")
            
            xs = contour[:, 0, 0]
            ys = contour[:, 0, 1]

            #if match < HuThreshold:
            #    plt.plot(xs, ys, linewidth=1.5)

            moments += (match, contour)

    print("moments:", moments[0::2])  

    hu_val = np.argmin(moments[0::2])

    if moments[0::2][hu_val] < HuThreshold:
        contour = moments[1::2][hu_val]

        m = cv2.moments(contour, binaryImage=True)
        m = cv2.HuMoments(m).flatten()

        with open(sys.argv[1] + "moments.txt", "wb") as f:
            pickle.dump(m, f)
        
        x, y, width, height = cv2.boundingRect(contour)

        cv2.drawContours(original, contour, -1, (0, 255, 0), 20)

        return original[y:y+height, x:x+width], (x, y, width, height)

def main():
    if "train" == sys.argv[1]:
        train()
        print("Moments trained")
        exit(0)
    if "itrain" in sys.argv[1]:
        train(True)
        print("Moments image-trained")
        exit(0)

    with open(sys.argv[2], "rb") as f:
        a4_moment = pickle.load(f)

    print("Reading", sys.argv[1])
    img = cv2.imread(sys.argv[1], 1)
    
    print("a4_moment hu:", a4_moment)
    qr_img, bounding_rect = hu_detect(a4_moment, img)

    # process QR
    #processQr(qr_code)

    plt.figure()
    plt.imshow(qr_img)
    plt.show()

if __name__ == "__main__":
    main()
