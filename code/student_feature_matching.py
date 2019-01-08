import numpy as np
import math


def match_features(features1, features2, x1, y1, x2, y2):

    Distance = np.zeros((features1.shape[0], features2.shape[0]))
    Value = []
    Hitx = []
    Hity = []
    for x in range(features1.shape[0]):

        for y in range(features2.shape[0]):

            #Extract the feature vector of image1

            ExtractedRow1 = features1[[x],:]

            #Extract the feature vector of image2

            ExtractedRow2 = features2[[y],:]

            #Calculate the Euclidean distance between the feature vectors and sort.

            SubtractedRow = ExtractedRow1 - ExtractedRow2

            Square = SubtractedRow*SubtractedRow
            Sum = Square.sum()

            Sum = math.sqrt(Sum)

            Distance[x,y] = Sum






        IndexPosition = np.argsort(Distance[x,:])

        #Take the two smallest distances between the feature vectors

        d1 = IndexPosition[0]
        d2 = IndexPosition[1]
        Position1 = Distance[x,d1]
        Position2 = Distance[x,d2]

        #Calculate the ratio of the two distances and check if it is above the threshold.
        ratio = Position1/Position2


        if ratio<0.8:  #Change to 0.9 while running Mount Rushmore

            Hitx.append(x)


            Hity.append(d1)

            Value.append(Position1)


    Xposition = np.asarray(Hitx)
    Yposition = np.asarray(Hity)
    matches = np.stack((Xposition,Yposition), axis = -1)
    confidences = np.asarray(Value)



    return matches, confidences

