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
	for point in range(len(best)):
		
		x = best[point][0]
		y = best[point][1]
		r = best[point][2]

		img1[x][y] = [0,0,255] 
	


	cv2.imwrite("Nbest.png", img1)
	return best

#CornerDetection()


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
def featuredescription():
    print('Enter Fetauredescription')
    image = cv2.imread('../Data/Train/Set1/1.jpg')
    # cv2.imwrite('img_feature.png', image)
    gray_feat = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_feat = np.float32(gray_feat)
    # 1st argument --> numbers ranging from 0 to 9,
    # 2nd argument, row = 2, col = 3
    arrayTest = np.array(CornerDetection())


    #np.random.randint(50, 100, size=(450, 2))  # This needs to be the ANMS output (x and y) of the best corners

    patch_size = 40

    r, c = arrayTest.shape  # Size of the ANMS
    # print('r,c', r, c)
    img_pad = np.pad(gray_feat, patch_size, 'constant',
                     constant_values=0)  # add a border around image for patching, zero to add black countering

    feat_desc = np.array(np.zeros((int((patch_size / 5) ** 2), 1)))

    epsilon = 10e-10

    for i in range(r):
        print('i for loop featue', i)

        # Patch around of the best ANMS
        patch_center = arrayTest[i]
        print('patccenter', patch_center)
        patch_x = int(patch_center[0] - patch_size / 2)
        print('x', patch_x)
        patch_y = int(patch_center[1] - patch_size / 2)
        print('y', patch_y)

        patch = img_pad[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        print('PATCH', patch)
        print('patchsize', patch.shape)

        # Apply Gauss blur
        blur_patch = cv2.GaussianBlur(patch, (5, 5), 0)
        print('patchBLUR', blur_patch)
        print('size BLURPATCH', blur_patch.shape)

        # Sub-sample to 8x8
        sub_sample = blur_patch[::5, ::5]
        print('Subsample0::5', sub_sample)
        print('sizesubsample', sub_sample.shape)
        cv2.imwrite('patch' + str(i) + '.png', sub_sample)

        # Re-sahpe to 64x1
        feats = sub_sample.reshape(int((patch_size / 5) ** 2), 1)
        print('Featsub_sampple,shape', feats)
        print('sizeFeats', feats.shape)

        # Make the mean 0
        feats = feats - np.mean(feats)
        print('feats', feats)

        # Make the variance 1
        feats = feats / (np.std(feats) + epsilon)
        cv2.imwrite('feature_vector' + str(i) + '.png', feats)
        print('featsVarince1', feats)
        feat_desc = np.dstack((feat_desc, feats))
        print('descr', feat_desc[0])
        print('reshape_64 -np.dstack', feat_desc)
        print('reshape_64 -np.dstack', feat_desc.shape)
    print('end features descri')

    return feat_desc[:, :, 1:]


featuredescription()


#Feature Matching



#input 4 feature pairs at random
#select at random 4 points to fit the model



def RANSAC(matches):

	N = 500


	for i in N:
		pair1, pair2 = randompairs(matches)
		H = homography(pair1, pair2)




def randompairs(matches):

	pairs = []
	for i in range(4):
		match = matches[i]
		pairs.append(pairs)
		img1 = []
		img2 = []
		for match in pairs:
			img1.append(match[0])
			img2.append(match[1])
		return img1, img2




#input: a list 4 selected random pairs for img1 and img2
def homography(img1_kp, img2_kp):
	A = []
	points = []

	for i in range(len(img1_kp)):
		x = img1_kp[i][0]
		y = img1_jp[i][1]

		u = img2_kp[i][0]
		v = img2_kp[i][1]

		A.append([x, y, 1, 0, 0, 0, (-u*x), (-u*y), -u])
		A.append([0, 0, 0, x, y, 1, (-v*x), (-v*y), -v])
		points = points + [u, v]

	#creates Array A
	A = np.array(A)
	H = np.linalg.lstsq(A, points)[0]
	zeros = np.zeros((8, 1))
	#H = zeros @ np.linalg.pinv(A) 
	H = np.reshape((np.concatenate((H, [1]), axis=-1)), (3, 3))
	#H = np.linalg.solve(A,zeros)
	print(H)

	return H

	#Do we transpose H?




#send pairs 
def inliers(matches,  H): #how do i know they are a pair

	Himg1 = H * img1 
	threshold = 3
	best = []



	for match in matches: 




		s = np.sum((img2-H)**2) #???

		if s < threshold:
			#inlier
			best.append()


'''

	num_inliers = 4
	s = 4 #minimum needed to fit the model (points)
	N = np.inf #number of samples
	count = 0
	e = 1.0

	while N > count:
		E = 1 - (num_inliers) / total_points

		if(E  < e):
			e = E

			N = numpy.log(1-p) / log(1-(1-e) ** s)

		count += 1'''







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
