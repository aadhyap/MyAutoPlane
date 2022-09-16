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
import copy

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

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    corners1 = CornerDetection(img1, img1_gray, 'orners1.png')
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

    anms_corners_img1 = ANMS(corners1, 300, img1, img1_gray, 'anms1.png')
    anms_corners_img2 = ANMS(corners2, 300, img2, img2_gray, 'anms2.png')

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

    anms_corners_img1 = np.delete(anms_corners_img1, 2, 1)
    anms_corners_img2 = np.delete(anms_corners_img2, 2, 1)
    patch_size = 40
    feat_desc = np.array(np.zeros((int((patch_size / 5) ** 2), 1)))
    epsilon = 10e-10

    feature1 = featuredescription(img1_gray, patch_size, anms_corners_img1, epsilon, feat_desc)
    feature2 = featuredescription(img2_gray, patch_size, anms_corners_img2, epsilon, feat_desc)

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

    ratioFM = 0.7

    cor1, cor2 = featureMatching(img1, img2, 'matching.png', feature1, feature2, anms_corners_img1, anms_corners_img2, ratioFM, epsilon)

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

    _,_,bestcorners1, bestcorners2 = RANSAC(cor1, cor2, img1, img2, 300, 'nransac.png')
    #Ho = RANSAC(cor1, cor2)
    #print('Hf', Ho)
    #print('pair1', rac1)
    #print('pair2', rac2

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """

    img1 = cv2.imread('../Data/Train/Set1/1.jpg')
    img2 = cv2.imread('../Data/Train/Set1/2.jpg')
    img3 = cv2.imread('../Data/Train/Set1/3.jpg')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = np.float32(img1_gray)
    img2_gray = np.float32(img2_gray)

    pts_src, pts_dst, _, _ = RANSAC(cor1, cor2, img1, img2, 300, 'ransac.png')
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(img1, h, (img2.shape[1], img2.shape[0]))
    cv2.imshow("Source Image", img1)
    cv2.imshow("Destination Image", img2)
    cv2.imshow("Warped Source Image", im_out)
    cv2.imwrite('Blending.png', im_out)
    #cv2.waitKey(0)


def CornerDetection(ori_image, gray_image, output_name):


    print('Called Corner Detection')

    corners = cv2.goodFeaturesToTrack(gray_image, 1000, 0.01, 10)
    corners = np.int0(corners)
    #print('shape', corners.shape)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(ori_image, (x, y), 3, (0,0,255), -1)

    cv2.imwrite(output_name, ori_image)
    return corners

def ANMS(stron_corners, best_num_corners, ori_image, img_gray, imagename):

    print('Called ANMS')
    l, _, w = stron_corners.shape
    r = np.inf * np.ones(l)
    best_coordinates = []
    anms_coordinates = []
    ED = 0
    #print('r', r)
    for i in range(l):
        coor1 = stron_corners[i]
        #print('coor1',coor1)
        xi = np.int(coor1[0][0])
        yi = np.int(coor1[0][1])
        for j in range(w):
            coor2 = stron_corners[j]
            xj = np.int(coor2[0][0])
            yj = np.int(coor2[0][1])
            if int(img_gray[yi, xi]) > int(img_gray[yj, xj]):
                ED = (xj - xi)**2 + (yj - yi)**2
            if ED < r[i]:
                r[i] = ED
                best_coordinates.append([xi, yi, r[i]])
    #print('que hice', best_coordinates)
    best_coordinates = np.array(best_coordinates, np.int0)
    #print('que hicex2', best_coordinates)
    #print('x2shpae', best_coordinates.shape)
    indices, dessorted = zip(*sorted(enumerate(-best_coordinates[:,2]), key=itemgetter(1)))
    #print('indi', indices)
    #print('sorted', dessorted)
    #indices = np.array(indices)
    #print('indicesarray', indices)
    #print('len', len(indices))

    for i in range(len(indices)):
        k = indices[i]
        #print('indice', k)
        anms_coordinates.append(best_coordinates[k])

    # print('finalmatrix', anms_coordinates)

    anms_coordinates = np.array(anms_coordinates)

    anms_coordinates = anms_coordinates[:best_num_corners, :]
    anms_image = copy.deepcopy(ori_image)

    for i in anms_coordinates:
        x,y,_ = i.ravel()
        cv2.circle(anms_image, (int(x), int(y)), 3, [0, 0, 255], -1)
    cv2.imwrite(imagename, anms_image)
    #print('best', anms_coordinates)
    return anms_coordinates

def featuredescription(image, patch_size, anmsPos, epsilon, feat_desc):
    print('Called Feature Description')
    anmsPos = np.array(anmsPos)
    r, c = anmsPos.shape  # Size of the ANMS
    img_pad = np.pad(image, 100, 'constant',
                     constant_values=0)# add a border around image for patching, zero to add black countering

    for i in range(r):

        patch_center = anmsPos[i]
        patch_x = abs(int(patch_center[0] - (patch_size / 2)))
        patch_y = abs(int(patch_center[1] - (patch_size / 2)))
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
    # ssd = []
    #print('shapeimag1', img1.shape) #img1(600x450), #img(450x600)
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

    # ploting the points in the image #img1(600x450), #imgshape(450x600)

    newImageshape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    newImage = np.zeros(newImageshape, type(img1.flat[0]))
    newImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    newImage[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    for i in range(len(matchpair_img1)):

        x1, y1 = matchpair_img1[i]
        x2, y2 = matchpair_img2[i]
        x2 = x2 + img1.shape[1]

        cv2.circle(newImage, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(newImage, (x2, y2), 3, (255, 0, 0), -1)
        cv2.line(newImage, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(imagename, newImage)
    return matchpair_img1, matchpair_img2

def RANSAC(pair1, pair2, img1, img2, N, imagename):# parametes is all img1 points matches and img2point matches

    print('Called RANSAC')
    bestcount = 0
    bestmatches = []
    threshold = 0.9
    n = len(pair1)
    for i in range(N):

        randomRow = np.random.randint(n, size=4)
        randomPair1 = np.array([pair1[randomRow[0], :], pair1[randomRow[1], :], pair1[randomRow[2], :], pair1[randomRow[3], :]])
        randomPair2 = np.array([pair2[randomRow[0], :], pair2[randomRow[1], :], pair2[randomRow[2], :], pair2[randomRow[3], :]])
        # print('random1', randomPair1)
        # print('random2', randomPair2)


        H = homography(randomPair1, randomPair2)
        #print('H', H)
        inliers = 0
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

        matchpair_img1 = np.array(bestmatches_img1)
        matchpair_img2 = np.array(bestmatches_img2)

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

    return randomPair1, randomPair2, bestmatches_img1, bestmatches_img2 #coordinates of the best matches


#input: a list 4 selected random pairs for img1 and img2
def homography(img1_kp, img2_kp):
    #print('called Homograhy')

    corners = np.float32(img1_kp)
    new_corners = np.float32(img2_kp)
    #print('ACornersHfloat', corners)
    #print('ANewCornersHfloat', new_corners)


    matrix = cv2.getPerspectiveTransform(corners, new_corners)
    #print('matrix in homography', matrix)
    # H = cv2.warpPerspective(H, (500, 600))
    # H = np.linalg.inv(matrix) # is this right? why do you need the inverse?
    return matrix


if __name__ == "__main__":
    main()
