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
from operator import itemgetter


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

    img1 = cv2.imread('../Data/Train/Set1/1.jpg')
    img2 = cv2.imread('../Data/Train/Set1/2.jpg')
    img3 = cv2.imread('../Data/Train/Set1/3.jpg')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = np.float32(img1_gray)
    img2_gray = np.float32(img2_gray)

    # cv2.imshow('original image', img1)
    # cv2.waitKey()

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    corners1 = CornerDetection(img1, img1_gray, 'corners1.png')
    corners2 = CornerDetection(img2, img2_gray, 'corners2.png')

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """

    img1 = cv2.imread('../Data/Train/Set1/1.jpg')
    img2 = cv2.imread('../Data/Train/Set1/2.jpg')
    img3 = cv2.imread('../Data/Train/Set1/3.jpg')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = np.float32(img1_gray)
    img2_gray = np.float32(img2_gray)

    best1 = ANMS(corners1, img1, 'anms1.png')
    best2 = ANMS(corners2, img2, 'anms2.png')

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """

    img1 = cv2.imread('../Data/Train/Set1/1.jpg')
    img2 = cv2.imread('../Data/Train/Set1/2.jpg')
    img3 = cv2.imread('../Data/Train/Set1/3.jpg')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = np.float32(img1_gray)
    img2_gray = np.float32(img2_gray)

    best1 = np.delete(best1, 2, 1)
    best2 = np.delete(best2, 2, 1)
    patch_size = 40
    feat_desc = np.array(np.zeros((int((patch_size / 5) ** 2), 1)))
    epsilon = 10e-10

    feature1 = featuredescription(img1_gray, patch_size, best1, epsilon, feat_desc)
    feature2 = featuredescription(img2_gray, patch_size, best2, epsilon, feat_desc)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """

    img1 = cv2.imread('../Data/Train/Set1/1.jpg')
    img2 = cv2.imread('../Data/Train/Set1/2.jpg')
    img3 = cv2.imread('../Data/Train/Set1/3.jpg')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = np.float32(img1_gray)
    img2_gray = np.float32(img2_gray)

    ratioFM = 0.99

    cor1, cor2 = featureMatching(img1, img2, 'matching.png', feature1, feature2, best1, best2, ratioFM, epsilon)

    """
    Refine: RANSAC, Estimate Homography
    """

    img1 = cv2.imread('../Data/Train/Set1/1.jpg')
    img2 = cv2.imread('../Data/Train/Set1/2.jpg')
    img3 = cv2.imread('../Data/Train/Set1/3.jpg')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = np.float32(img1_gray)
    img2_gray = np.float32(img2_gray)

    #rac1, rac2 = RANSAC(cor1, cor2)
    Ho = RANSAC(cor1, cor2)
    print('Hf', Ho)
    #print('pair1', rac1)
    #print('pair2', rac2)



    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """






def CornerDetection(ori_image, gray_image, output_name):

    print('Called Corner Detection')

    dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    ori_image[dst > 0.01 * dst.max()]=[0, 0, 255]
    cv2.imwrite(output_name, ori_image)

    return dst

def ANMS(cornerScoreImage, ori_image, imagename):

    print("Called ANMS")

    radius = np.inf
    distance = np.inf
    all_r = []
    strong = []
    size = len(cornerScoreImage)

    for i in range(size):
        for j in range(size):
            if (cornerScoreImage[i][j] > 0.01 * cornerScoreImage.max()):  # set everything lower than threshold to 0
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
    all_r.sort(key=lambda x: x[2], reverse=True)
    best = all_r[:100]
    best = np.array(best)

    for point in range(len(best)):

        x = best[point][0]
        y = best[point][1]
        r = best[point][2]
        finalimage = cv2.circle(ori_image, (x, y), 3, 255, -1)

    cv2.imwrite(imagename, finalimage)

    return best

def featuredescription(image, patch_size, anmsPos, epsilon, feat_desc):
    print('Called Feature Description')

    r, c = anmsPos.shape  # Size of the ANMS
    img_pad = np.pad(image, patch_size, 'constant',
                     constant_values=0)  # add a border around image for patching, zero to add black countering

    for i in range(r):
        patch_center = anmsPos[i]
        patch_x = abs(int(patch_center[0] - patch_size / 2))
        patch_y = abs(int(patch_center[1] - patch_size / 2))
        #ac print('y', patch_y) # Here there is a problem with the axis Y for image 2 when the value is less than 20.
        patch = img_pad[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]

        # Apply Gauss blur
        blur_patch = cv2.GaussianBlur(patch, (5, 5), 0)

        # Sub-sample to 8x8
        sub_sample = blur_patch[::5, ::5]
        cv2.imwrite('./patch/patch' + str(i) + '.png', sub_sample)

        # Re-sahpe to 64x1
        feats = sub_sample.reshape(int((patch_size / 5) ** 2), 1)

        # Make the mean 0
        feats = feats - np.mean(feats)

        # Make the variance 1
        feats = feats / (np.std(feats) + epsilon)
        cv2.imwrite('./feature/feature_vector' + str(i) + '.png', feats)

        feat_desc = np.dstack((feat_desc, feats))

    return feat_desc[:, :, 1:]

def featureMatching(img1, img2, imagename, featv1, featv2, best_corners1, best_corners2, ratioFM, epsilon):
    print('Called Feature Matching')
    matchpair_img1 = []
    matchpair_img2 = []

    r, c, s = featv1.shape
    h, v, d = featv2.shape

    for i in range(s):
        ssd = []
        for j in range(d):
            ssd.append(sum(pow(featv1[:, :, i] - featv2[:, :, j], 2)))
        indices, ssdsorted = zip(*sorted(enumerate(ssd), key=itemgetter(1)))
        ratio = ssdsorted[0] / (ssdsorted[1] + epsilon)

        if ratio < ratioFM:
            matchpair_img1.append(best_corners1[indices[0]])
            matchpair_img2.append(best_corners2[indices[0]])

    matchpair_img1 = np.array(matchpair_img1)
    matchpair_img2 = np.array(matchpair_img2)

    # ploting the points in the image
    newImageshape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    newImage = np.zeros(newImageshape, type(img1.flat[0]))
    newImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    newImage[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    for i in range(len(matchpair_img1)):

        x1, y1 = matchpair_img1[i]
        x2, y2 = matchpair_img2[i]
        x2 = x2 + img1.shape[1]

        cv2.circle(newImage, (x1, y1), 3, 255, 1)
        cv2.circle(newImage, (x2, y2), 3, 255, 1)
        cv2.line(newImage, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(imagename, newImage)
    return matchpair_img1, matchpair_img2

def RANSAC(pair1, pair2):# parametes is all img1 points matches and img2point matches

    print('Called RANSAC')
    # pick rnadom x and y
    n = len(pair1)
    randomRow = np.random.randint(n, size=4)
    randomPair1 = np.array([pair1[randomRow[0], :], pair1[randomRow[1], :], pair1[randomRow[2], :], pair1[randomRow[3], :]])
    randomPair2 = np.array([pair2[randomRow[0], :], pair2[randomRow[1], :], pair2[randomRow[2], :], pair2[randomRow[3], :]])
    # print('random1', randomPair1)
    # print('random2', randomPair2)

    H = homography(randomPair1, randomPair2)

    return H
''' bestcount = 0
    bestmatches = []
    threshold = 9


    #for i in N:

        # pair1, pair2 = randompairs(matches) #delete this line you don't need random pairs function. Uses img1points and img2points instead
    H = homography(randomPair1, randomPair2)
    fordot = np.concatenate((randomPair1, np.ones((randomPair1.shape[0], 1))), axis = 1 )
    img2_new = np.dot(H, fordot.T )
    img2_new[-1, :] = img2_new[-1, :] + 0.000001
    img2_new = img2_new / (img2_new[-1 :])
    img2_new = img2_new.T

    x = img2_new[:, 1]
    y = img2_new[:, 0]


    img2_new = np.stack((y,x), axis = -1)
    ssd = np.sum(((randomPair2 - img2_new)**2), axis = 1)

    inliner = np.where(ssd < threshold)
    if(len(inliner[0]) > bestcount or bestcount ==0 ):
        bestcount = len(inliner[0])
        bestmatches_img1 = randomPair1[inliner]
        bestmatches_img2 = randomPair2[inliner]


    return bestmatches_img1, bestmatches_img2 #coordinates of the best matches'''


''' def randompairs(matches):

    pairs = []
    for i in range(4):
        match = matches[i]
        pairs.append(pairs)
        img1 = []
        img2 = []
        for match in pairs:
            img1.append(match[0])
            img2.append(match[1])
        return img1, img2'''

#input: a list 4 selected random pairs for img1 and img2
def homography(img1_kp, img2_kp):
    print('called Homograhy')

    corners = np.float32(img1_kp)
    new_corners = np.float32(img2_kp)
    print('ACornersHfloat', corners)
    print('ANewCornersHfloat', new_corners)


    matrix = cv2.getPerspectiveTransform(corners, new_corners)
    print('matrix in homography', matrix)
    # H = cv2.warpPerspective(H, (500, 600))
    # H = np.linalg.inv(matrix) # is this right? why do you need the inverse?
    return matrix


if __name__ == "__main__":
    main()
