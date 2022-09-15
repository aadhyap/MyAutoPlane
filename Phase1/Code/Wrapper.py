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

    newim = featureMatching(img1, img2, 'matching.png', feature1, feature2, best1, best2, ratioFM, epsilon)
    #cor1, cor2 = featureMatching(feature1, feature2, best1_array, best2_array, ratioFM, epsilon)


def CornerDetection(ori_image, gray_image, output_name):

    print('Called Corner Detection')

    dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    # print('dstinfun', dst)

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
            # peak local max
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
    # print('ANMSinfun', best)
    # print('ANMSshapeinfun', best.shape)
    for point in range(len(best)):
        # print('enter i best')
        x = best[point][0]
        # print('x', x)
        y = best[point][1]
        # print('y', y)
        r = best[point][2]
        # print('r', r)
        finalimage = cv2.circle(ori_image, (x, y), 3, 255, -1)

    cv2.imwrite(imagename, finalimage)

    return best

def featuredescription(image, patch_size, anmsPos, epsilon, feat_desc):
    print('Called Feature Description')

    r, c = anmsPos.shape  # Size of the ANMS
    img_pad = np.pad(image, patch_size, 'constant',
                     constant_values=0)  # add a border around image for patching, zero to add black countering
    # cv2.imwrite('imagepad.png', img_pad)

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
        # print('feature in the for loop', i)
        ssd = []
        for j in range(d):
            ssd.append(sum(pow(featv1[:, :, i] - featv2[:, :, j], 2)))
        indices, ssdsorted = zip(*sorted(enumerate(ssd), key=itemgetter(1)))
        ratio = ssdsorted[0] / (ssdsorted[1] + epsilon)
        # print('ratio', ratio)
        # print('ssd, indx, sorted', ssd, ssdsorted, indices)
        if ratio < ratioFM:
            matchpair_img1.append(best_corners1[indices[0]])
            matchpair_img2.append(best_corners2[indices[0]])
            # print('matchpair1', matchpair_img1)
            # print('matchpair2', matchpair_img2)
    # print('match pair final1', matchpair_img1)
    # print('match pair final2', matchpair_img2)
    matchpair_img1 = np.array(matchpair_img1)
    matchpair_img2 = np.array(matchpair_img2)
    # print('matc1', matchpair_img1)
    # print('matc2', matchpair_img2)
    # print('lenmatc1', len(matchpair_img1))
    # print('lenmatc2', len(matchpair_img2))
    # print('matc1shape', matchpair_img1.shape)
    # print('matc2shape', matchpair_img2.shape)

    # ploting the points in the image
    newImageshape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    newImage = np.zeros(newImageshape, type(img1.flat[0]))
    newImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    newImage[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    for i in range(len(matchpair_img1)):
        # print('enter i draw', i)
        x1, y1 = matchpair_img1[i]
        x2, y2 = matchpair_img2[i]
        x2 = x2 + img1.shape[1]

        cv2.line(newImage, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(newImage, (x1, y1), 3, 255, 1)
        cv2.circle(newImage, (x2, y2), 3, 255, 1)

    return cv2.imwrite(imagename, newImage)



if __name__ == "__main__":
    main()
