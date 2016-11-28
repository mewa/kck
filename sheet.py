# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle
import skimage.io

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
        pickle.dump(sum / n, f, protocol=2)

def hu_diff(mom1, mom2):
    tmp1 = []
    tmp2 = []
    for i,_ in enumerate(mom1):
        if mom1[i] != 0 and mom2[i] != 0:
            tmp1.append(mom1[i])
            tmp2.append(mom2[i])
    mom1 = tmp1
    mom2 = tmp2
    mom1 = -np.sign(mom1) * np.log10(np.abs(mom1))
    mom2 = -np.sign(mom2) * np.log10(np.abs(mom2))

    return np.max(np.abs((mom1 - mom2) / mom2))

def hu_detect(moment, img, min_coverage=0.1, HuThreshold=2.5, invert=False, adaptive=True, dilation_level=2, dilation_kernel=(3, 3), max_coverage=1):
    original = img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (15, 15), 0)

    blurred = img
    if invert:
        flag = cv2.THRESH_BINARY_INV
    else:
        flag = cv2.THRESH_BINARY
    ret, img = cv2.threshold(img, 0, 255, flag + cv2.THRESH_OTSU)

    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones( dilation_kernel ), iterations = dilation_level)

    if adaptive:
        b = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
        b = cv2.morphologyEx(b, cv2.MORPH_ERODE, np.ones( (3, 3) ), iterations = 2)
        cv2.subtract(img, b, img)

    # show img as found before finding contours
    plt.figure()
    plt.imshow(img)

    # uncomment to save moments
    #with open("moments.txt", "wb") as f:
    #    pickle.dump(mom, f)

    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    moments = []

    for contour in contours:
        contour_moment = cv2.moments(contour, binaryImage=True)
        area = contour_moment['m00']
        coverage = area / (img.shape[0] * img.shape[1])

        contour_moment = cv2.HuMoments(contour_moment).flatten()

        if coverage > min_coverage:
            match = hu_diff(moment, contour_moment)#cv2.matchShapes(moment, contour, 3, 0.0)
            print("coverage", coverage)
            print("moment", contour_moment)
            print("-" * 15)
            print("match:", match)
            print("-" * 15)

            xs = contour[:, 0, 0]
            ys = contour[:, 0, 1]

            #if match < HuThreshold:
            #    plt.plot(xs, ys, linewidth=1.5)

            moments += (match, contour)

    print("moments:", moments[0::2])
    plt.figure()
    plt.imshow(img)
    if (len(moments) == 0):
        plt.show()
        exit(1)
    hu_val = np.argmin(moments[0::2])

    if moments[0::2][hu_val] < HuThreshold:
        contour = moments[1::2][hu_val]

        m = cv2.moments(contour, binaryImage=True)
        m = cv2.HuMoments(m).flatten()

        with open(sys.argv[1] + "moments.txt", "wb") as f:
            pickle.dump(m, f)

        x, y, width, height = cv2.boundingRect(contour)

        #cv2.drawContours(original, contour, -1, (0, 255, 0), 20)
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        c = contour
        c[:, 0, 0] -= x
        c[:, 0, 1] -= y
        cv2.fillPoly(mask, pts=[c], color=(255,255,255))

        cropped = original[y:y+height, x:x+width]
        cropped = cv2.bitwise_and(cropped, mask)

        return cropped, (x, y, width, height), contour
    return original, (0, 0, original.shape[1], original.shape[0]), 0

def draw_plot(image):
    plt.figure()
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

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

    with open(sys.argv[3], "rb") as f:
        qr_moment = pickle.load(f)

    print("Reading", sys.argv[1])
    img = cv2.imread(sys.argv[1], 1)

    print("page_moment hu:", a4_moment)
    print("qr_moment hu:", a4_moment)

    original = img

    img, bounding_rect, _ = hu_detect(a4_moment, img)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones( (13, 13) ))

    img, bounding_rect2, contour = hu_detect(qr_moment, img, HuThreshold=40, min_coverage=0.025, max_coverage=1, invert=True, adaptive=False, dilation_level=3, dilation_kernel=(9,9))

    logo = cv2.imread('logo.png', -1)

    x, y, _, _ = bounding_rect

    #for c in range(0,3):
    #original[y:y+logo.shape[0], x:x+logo.shape[1], c] = logo[:,:,c] * (logo[:,:,3]/255.0) +  original[y:y+logo.shape[0], x:x+logo.shape[1], c] * (1.0 - logo[:,:,3]/255.0)

    if bounding_rect2 != None:
        x2, y2, width, height = bounding_rect2
        coverage = (float(width*height) / float(logo.shape[0]*logo.shape[1]))*5
        logo = cv2.resize(logo, (int(coverage * logo.shape[1]), int(coverage * logo.shape[0])), interpolation = cv2.INTER_CUBIC)
        x += x2
        y += y2

    #original[y:y+height, x:x+width] = 0
    contour[:, 0, 0] += x
    contour[:, 0, 1] += y
    cv2.fillPoly(original, pts=[contour], color=(0xec, 0x47, 0x7a))

    draw_plot(original)


if __name__ == "__main__":
    main()
