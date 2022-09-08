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


def ANMS(cornerScoreImage):
	print("CALLED ANMS")

	print("THIS IS ALL THE OG SCORES")
	print(cornerScoreImage)

	radius = np.inf 
	distance = np.inf
	size = len(cornerScoreImage)
	all_r = []

	strong = []
	for i in range(size):
		for j in range(size):

			#peak local max
			if(cornerScoreImage[i][j] > 0.01*cornerScoreImage.max()): #set everything lower than threshold to 0
				corner = [i, j, cornerScoreImage[i][j]]
				strong.append(corner)
			else:
				cornerScoreImage[i][j] = -np.inf
	print("ALL THE STRONGS ", len(strong))
	print(strong)


	radius = [np.inf] * len(strong)
	for n in range(len(strong)):
		x = strong[n][0]
		y = strong[n][1]
		score = strong[n][2]

		for m in range(len(strong)):


			if(m != n):
				i = strong[m][0]
				j = strong[m][1]
				compare_score = strong[m][2]

				if(score > compare_score):
					distance = ((x - i) ** 2) + ((y-j)**2)
				if(distance < radius[n]):
					radius[n] = distance
		coor = [x,y,radius[n]]
		all_r.append(coor)
	print("ALL THE RADIUSES ")
	print(all_r)

	'''for x in range(size):
		for y in range(size):
			if(cornerScoreImage[x][y] != 0):

				#the next N strong corner

						
				for i in range(size):
					for j in range(size):

						if(cornerScoreImage[i][j] != 0 and cornerScoreImage[x][y] > cornerScoreImage[i][j] and (x != i and y != j)):
							distance = ((x - i) ** 2) + ((y-j)**2)

						if(distance < radius):
							radius = distance


				coor = [x,y,radius]
				all_r.append(coor)'''


	all_r.sort(key=lambda x:x[2], reverse=True)



	return (all_r[:100])



def CornerDetection():
	img1 = cv2.imread('../Data/Train/Set1/1.jpg')
	img2 = cv2.imread('../Data/Train/Set1/2.jpg')
	img3 = cv2.imread('../Data/Train/Set1/3.jpg')


	gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	
	dst = cv2.cornerHarris(gray,2,3,0.04)
	
	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	best = ANMS(dst)
	print("BEST")
	print(best)
	for point in range(len(best)):
		
		x = best[point][0]
		y = best[point][1]
		r = best[point][2]
	



		
		img1[x][y] = [0,0,255] 
	


	cv2.imwrite("Nbest.png", img1)

CornerDetection()


"""
Perform ANMS: Adaptive Non-Maximal Suppression
Save ANMS output as anms.png
"""
#Input: Image of the cornerscore 



#first it needs a corner Nstrong coordinate

#Then it needs to score of the Nstrong coordinate

#Then feed it to the the thing 




										#every 5th row and 42nd  
										#gaussian blur after 

			

			#add episolon 
			#down sample to get more points the two points are not as close
			#blending do same intentsity

			#sprt r in descending order
			#panaroma stiching find the homography matrix
			#take inverse than the homography then apply i(original image, inverse pertreb it, then inverse of that thing, take the normal )


#superglue magic leap read the paper
#take patch of image, w and h
#take the inveerse and width and height
#take a white square in the middle of a black square at the end you should have a white square
#cv2.warpperspective, don't use it bacause it's not differentiable instead use cornea
#Take the average of the supervised, use supervised P(a) which is the patch and P(b) has to be similirarity (P(a) - P(b)) <-- loss unsupervised
#use that loss to backpropogate, use cornea. 
#spatial transform network (STN)


#Tesla uses pure math
"""
Feature Descriptors
Save Feature Descriptor output as FD.png
"""

"""
Feature Matching
Save Feature Matching output as matching.png
"""

#input image 1 point and sum of all image 2 points ("are they talking about features?")
def FeatureMatching(kp1, kp2):
	img1 = cv2.imread('../Data/Train/Set1/1.jpg')
	img2 = cv2.imread('../Data/Train/Set1/2.jpg')

	# Initiate ORB detector
	orb = cv2.ORB_create()

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)
	

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)


	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)


	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	cv2.imwrite("Matches.png", img3)


def match(kp1, kp2):
	matches = []

	#for point in kp1:




#FeatureMatching()



"""
Refine: RANSAC, Estimate Homography
"""

"""
Image Warping + Blending
Save Panorama output as mypano.png
"""


if __name__ == "__main__":
    main()
