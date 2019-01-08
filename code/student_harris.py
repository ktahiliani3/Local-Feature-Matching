import cv2
import numpy as np
import math

def ANMS (x , y, r, maximum):



    #x is an array of length N
    #y is an array of length N
    #r is the cornerness score
    #max is the no of corners that are required

    i = 0
    j = 0
    NewList = []

    while i < len(x):

        minimum = 1000000000000 #random large value

        FirstCoordinate, SecondCoordinate = x[i], y[i]

        while j < len(x):

            CompareCoordinate1, CompareCoordinate2 = x[j], y[j]

            if (FirstCoordinate != CompareCoordinate1 and SecondCoordinate != CompareCoordinate2) and r[i] < r[j]:

                distance = math.sqrt((CompareCoordinate1 - FirstCoordinate)**2 + (CompareCoordinate2 - SecondCoordinate)**2)

                if distance < minimum:

                    minimum = distance

            j = j + 1

        NewList.append([FirstCoordinate, SecondCoordinate, minimum])

        i = i + 1
        j = 0

    NewList.sort(key = lambda t: t[2])

    NewList = NewList[len(NewList)-maximum:len(NewList)]

    return NewList




def get_interest_points(image, feature_width):



    alpha = 0.04
    threshold = 10000


    XCorners = []
    YCorners = []
    RValues = []

    #Compute the size of the image.

    ImageRows = image.shape[0]
    ImageColumns = image.shape[1]

    #Use the soble filter to calculate the x and y derivative of the image

    Xderivative = cv2.Sobel(image, cv2.CV_64F,1,0,ksize=5)
    Yderivative = cv2.Sobel(image, cv2.CV_64F,0,1,ksize=5)


    #Define matrices Ixx, Iyy and Ixy

    Ixx = (Xderivative)*(Xderivative)
    Iyy = (Yderivative)*(Yderivative)
    Ixy = (Xderivative)*(Yderivative)

    #loop over the image to compute cornerness score of each pixel

    for i in range(16, ImageRows - 16):
        for j in range(16, ImageColumns - 16):

            Ixx1 = Ixx[i-1:i+1, j-1:j+1]
            Iyy1 = Iyy[i-1:i+1, j-1:j+1]
            Ixy1 = Ixy[i-1:i+1, j-1:j+1]

            Ixxsum = Ixx1.sum()
            Iyysum = Iyy1.sum()
            Ixysum = Ixy1.sum()

            Determinant = Ixxsum*Iyysum - Ixysum**2
            Trace = Ixxsum + Iyysum
            R = Determinant - alpha*(Trace**2)

            #Check if the cornerness score is above the threshold and if the pixel is an eligible corner pixel

            if R > threshold:

                XCorners.append(j)
                YCorners.append(i)
                RValues.append(R)


    XCorners = np.asarray(XCorners)
    YCorners = np.asarray(YCorners)
    RValues = np.asarray(RValues)

    #Use ANMS to evenly distribute the corners in the image.

    NewCorners = ANMS(XCorners, YCorners, RValues, 3025)

    NewCorners = np.asarray(NewCorners)


    #Return the x-y coordinates and cornerness score of the eligible corners.

    x = NewCorners[:,0]
    y = NewCorners[:,1]
    scales = NewCorners[:,2]


    return x,y, scales


