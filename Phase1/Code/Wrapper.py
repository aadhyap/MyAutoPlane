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







	'''Nstrong =cv2.goodFeaturesToTrack(gray,1000,0.000000001,10)
	corners = np.int0(Nstrong)
	print("CORNERS")
	print(corners)
	for i in corners:
	    x,y = i.ravel()
	    cv2.circle(img1,(x,y),3,255,-1)'''





	gray = np.float32(gray)
	print("length of image ", len(gray))

	print("IMAGE ")
	print("IMAGE 1")
	dst = cv2.cornerHarris(gray,2,3,0.04)
	print("length of corner harris ", len(dst))


	print("CORNER HARRIS")
	print(dst)
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)
	print("length of dilation ", len(dst))
	print("DILATED ")
	print(dst)
	# Threshold for an optimal value, it may vary depending on the image.
	corners = [] 

	'''for i in range(len(dst)):
		for j in range(len(dst)):
	'''
	img1[dst>0.01*dst.max()]=[0,0,255] #prints True or False values if it's above 0.01 than 1% percent more confidence than the highest confidence

	print("THRESHOLD APPLIED")
	print(img1)

	print("MAX ")
	print(dst>0.01*dst.max())
	
	cv2.imshow('dst',img1)
	cv2.waitKey(5000)

	cv2.imwrite("corner.png", img1)

CornerDetection()

"""
Perform ANMS: Adaptive Non-Maximal Suppression
Save ANMS output as anms.png
"""
#Input: Image of the cornerscore 



#first it needs a corner Nstrong coordinate

#Then it needs to score of the Nstrong coordinate

#Then feed it to the the thing 
def ANMS(cornerScoreImage):

	radius = np.inf 
	distance = np.inf
	size = len(cornerScoreImage)
	for i in range(size):
		for j in range(size):
			#peak local max
			if(cornerScoreImage[i][j] < 0.01*cornerScoreImage.max()): #set everything lower than threshold to 0
				cornerScoreImage[i][j] = 0 


				for x in range(size):
					for y in range(size):
						if(cornerScoreImage[x][y] != 0):
							#the next N strong corner

						
							for i in range(size):
								for j in range(size):
									if(cornerScoreImage[i][j] != 0 and cornerScoreImage[x][y] > cornerScoreImage[i][j]):
										distance = ((x - i) ** 2) + ((y-j)**2)

									if(distance < radius):
										r = distance

			#add episolon 
			#down sample to get more points the two points are not as close




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
