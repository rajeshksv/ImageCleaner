# import the necessary packages
# Author: Richa Agrawal (with no help especially from Rajesh KSV)
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
import os,fnmatch

def getImageScore(img1, img2, showPlot):
    # Initiaite ORB and find descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 12, key_size = 20, multi_probe_level = 2)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    nMatches = 0
    for i,(m_n) in enumerate(matches):
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            nMatches = nMatches+1
            
    #print nMatches   # Remove this later

    # Draw matches.
    if(showPlot):
        draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        plt.imshow(img3),plt.show()
    return nMatches


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", help="first input image")
ap.add_argument("-s", "--second", help="second")
ap.add_argument("--folder", help="folder with images")
ap.add_argument("--threshold", help="Threshold for identifying duplicates")
args = vars(ap.parse_args())

def convertToGreyScale(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

threshold = 100
if args["threshold"] is not None:
    threshold = int(args["threshold"])

duplicateList = []

# load the two input images
if args["first"] is not None:
    img1 = convertToGreyScale(args["first"])
    img2 = convertToGreyScale(args["second"])
    score = getImageScore(img1, img2, 1)
    print "Score: " + str(score)

# img1 comes from the first image and we cycle through the second image
elif args["folder"] is not None:
    folder = args["folder"]
    files = fnmatch.filter(os.listdir(folder), '*.JPG')
    index1 = 0
    while (index1 < len(files)):
        file1 = files[index1]
        duplicateList.append(file1)
        for index2 in range(index1+1, len(files)):
            file2 = files[index2]
            img1 = convertToGreyScale(os.path.join(folder,file1))
            img2 = convertToGreyScale(os.path.join(folder,file2))
            score = getImageScore(img1, img2, 0)
            if (score > threshold):
                duplicateList.append(file2)
            else:
                if len(duplicateList) > 1:
                    print "Duplicate images found " + ', '.join(duplicateList)
                duplicateList = []
                index1 = index2
                break
        if (index2 == len(files) - 1):
            if (len(duplicateList) > 1):
                print "Duplicate images found " + ', '.join(duplicateList)
            break
else:
    print "No arguments provided. Provide folder or two images"
