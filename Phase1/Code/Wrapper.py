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

            # peak local max
            if (cornerScoreImage[i][j] > 0.01 * cornerScoreImage.max()):  # set everything lower than threshold to 0
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

            if (m != n):
                i = strong[m][0]
                j = strong[m][1]
                compare_score = strong[m][2]

                if (score > compare_score):
                    distance = ((x - i) ** 2) + ((y - j) ** 2)
                if (distance < radius[n]):
                    radius[n] = distance
        coor = [x, y, radius[n]]
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

    all_r.sort(key=lambda x: x[2], reverse=True)
    print('allr', all_r[:100])

    return (all_r[:100])


def CornerDetection():
    img1 = cv2.imread('../Data/Train/Set1/1.jpg')
    img2 = cv2.imread('../Data/Train/Set1/2.jpg')
    img3 = cv2.imread('../Data/Train/Set1/3.jpg')

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    print('output corner Harris', dst)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    print('output dilate', dst)

    best = ANMS(dst)
    print('best after ANMS', best)

    print("PUUTTING ON IMAGE ", len(best))
    for point in best:

        x = point[0]
        y = point[1]
        r = point[2]

        img1[x][y] = [0, 0, 255]
        for j in best:
            if ((((x - j[0]) ** 2) + ((y - j[1]) ** 2)) < r):
                best.remove(j)

    cv2.imwrite("Nbest.png", img1)


CornerDetection()

"""
Perform ANMS: Adaptive Non-Maximal Suppression
Save ANMS output as anms.png
"""
# Input: Image of the cornerscore


# first it needs a corner Nstrong coordinate

# Then it needs to score of the Nstrong coordinate

# Then feed it to the the thing


# every 5th row and 42nd
# gaussian blur after


# add episolon
# down sample to get more points the two points are not as close
# blending do same intentsity

# sprt r in descending order
# panaroma stiching find the homography matrix
# take inverse than the homography then apply i(original image, inverse pertreb it, then inverse of that thing, take the normal )


# superglue magic leap read the paper
# take patch of image, w and h
# take the inveerse and width and height
# take a white square in the middle of a black square at the end you should have a white square
# cv2.warpperspective, don't use it bacause it's not differentiable instead use cornea
# Take the average of the supervised, use supervised P(a) which is the patch and P(b) has to be similirarity (P(a) - P(b)) <-- loss unsupervised
# use that loss to backpropogate, use cornea.
# spatial transform network (STN)


# Tesla uses pure math
"""
Feature Descriptors
Save Feature Descriptor output as FD.png
"""


def featuredescription():
    print('Enter Fetauredescription')
    image = cv2.imread('../Data/Train/Set1/1.jpg')
    # cv2.imwrite('img_feature.png', image)
    gray_feat = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_feat = np.float32(gray_feat)
    # 1st argument --> numbers ranging from 0 to 9,
    # 2nd argument, row = 2, col = 3
    arrayTest = np.array([[332, 43, 62984], [332, 44, 62485], [331, 44, 62442], [330, 44, 62401], [332, 45, 61988], [331, 45, 61945], [330, 45, 61904], [330, 43, 59105], [331, 43, 58868], [145, 305, 26713], [145, 304, 26690], [145, 303, 26669], [146, 305, 26388], [146, 304, 26365], [146, 303, 26344], [147, 305, 26065], [147, 304, 26042], [147, 303, 26021], [310, 293, 20617], [308, 293, 19962], [309, 293, 19700], [442, 285, 16562], [442, 286, 16465], [441, 285, 16325], [441, 286, 16228], [440, 285, 16090], [440, 286, 15993], [246, 134, 14977], [440, 284, 14965], [247, 134, 14810], [246, 133, 14800], [248, 134, 14645], [247, 133, 14633], [246, 132, 14625], [248, 133, 14468], [247, 132, 14458], [248, 132, 14293], [175, 220, 7673], [174, 220, 7618], [173, 220, 7565], [245, 26, 7514], [175, 221, 7508], [245, 27, 7481], [174, 221, 7453], [245, 28, 7450], [151, 391, 7412], [150, 391, 7405], [149, 391, 7400], [173, 221, 7400], [175, 222, 7345], [246, 26, 7345], [246, 27, 7312], [406, 449, 7300], [174, 222, 7290], [246, 28, 7281], [405, 449, 7241], [173, 222, 7237], [404, 449, 7184], [247, 26, 7178], [247, 27, 7145], [247, 28, 7114], [390, 146, 5440], [52, 288, 5045], [52, 289, 4904], [52, 290, 4765], [88, 363, 4201], [88, 362, 4122], [88, 361, 4045], [192, 151, 3205], [436, 401, 3204], [330, 394, 3172], [435, 401, 3145], [329, 394, 3141], [328, 394, 3112], [436, 402, 3109], [434, 401, 3088], [330, 393, 3065], [435, 402, 3050], [329, 393, 3034], [436, 403, 3016], [296, 202, 3005], [328, 393, 3005], [434, 402, 2993], [330, 392, 2960], [435, 403, 2957], [257, 422, 2952], [329, 392, 2929], [296, 203, 2900], [328, 392, 2900], [434, 403, 2900], [258, 422, 2845], [438, 231, 2813], [296, 204, 2797], [376, 367, 2741], [259, 422, 2740], [376, 368, 2692], [376, 369, 2645], [216, 449, 2341], [380, 49, 2320], [380, 48, 2313]])

    # np.random.randint(50, 100,
                #                  size=(450, 2))  # This needs to be the ANMS output (x and y) of the best corners

    patch_size = 40

    r, c = arrayTest.shape  # Size of the ANMS
    # print('r,c', r, c)
    img_pad = np.pad(gray_feat, patch_size, 'constant',
                     constant_values=0)  # add a border around image for patching, zero to add black countering

    feat_desc = np.array(np.zeros((int((patch_size / 5) ** 2), 1)))

    epsilon = 10e-10

    for i in range(r):
        # print('i for loop featue', i)

        # Patch around of the best ANMS
        patch_center = arrayTest[i]
        # print('patccenter', patch_center)
        patch_x = int(patch_center[0] - patch_size / 2)
        # print('x', patch_x)
        patch_y = int(patch_center[1] - patch_size / 2)
        # print('y', patch_y)

        patch = img_pad[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
        # print('PATCH', patch)
        # print('patchsize', patch.shape)

        # Apply Gauss blur
        blur_patch = cv2.GaussianBlur(patch, (5, 5), 0)
        # print('patchBLUR', blur_patch)
        # print('size BLURPATCH', blur_patch.shape)

        # Sub-sample to 8x8
        sub_sample = blur_patch[::5, ::5]
        # print('Subsample0::5', sub_sample)
        # print('sizesubsample', sub_sample.shape)
        cv2.imwrite('patch' + str(i) + '.png', sub_sample)

        # Re-sahpe to 64x1
        feats = sub_sample.reshape(int((patch_size / 5) ** 2), 1)
        # print('Featsub_sampple,shape', feats)
        # print('sizeFeats', feats.shape)

        # Make the mean 0
        feats = feats - np.mean(feats)
        # print('feats', feats)

        # Make the variance 1
        feats = feats / (np.std(feats) + epsilon)
        cv2.imwrite('feature_vector' + str(i) + '.png', feats)
        # print('featsVarince1', feats)
        feat_desc = np.dstack((feat_desc, feats))
        # print('descr', feat_desc[0])
        # print('reshape_64 -np.dstack', feat_desc)
        # print('reshape_64 -np.dstack', feat_desc.shape)
    print('end features descri')

    return feat_desc[:, :, 1:]


featuredescription()

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
