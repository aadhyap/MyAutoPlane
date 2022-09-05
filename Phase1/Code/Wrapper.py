#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""









def CornerDetection():
	img1 = cv2.imread('../Data/Train/Set1/1.jpg')
	img2 = cv2.imread('../Data/Train/Set1/2.jpg')
	img3 = cv2.imread('../Data/Train/Set1/3.jpg')


	gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	cv2.imwrite("gray.png", gray)
	gray = np.float32(gray)





	Nstrong =cv2.goodFeaturesToTrack(gray,1000,0.000000001,10)
	corners = np.int0(Nstrong)
	print("CORNERS")
	print(corners)
	for i in corners:
	    x,y = i.ravel()
	    cv2.circle(img1,(x,y),3,255,-1)



	#cv2.imwrite("Harris.png", img1)
	cv2.imwrite("goodFeaturesToTrack.png", img1)

CornerDetection()

"""
Perform ANMS: Adaptive Non-Maximal Suppression
Save ANMS output as anms.png
"""

"""
Feature Descriptors
Save Feature Descriptor output as FD.png
"""

"""
Feature Matching
Save Feature Matching output as matching.png
"""

"""
Refine: RANSAC, Estimate Homography
"""

"""
Image Warping + Blending
Save Panorama output as mypano.png
"""


if __name__ == "__main__":
    main()
