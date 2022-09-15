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
import random

# Add any python libraries here


def featuredescription(image, patch_size, anmsPos, epsilon, feat_desc):
    print('Called Feature Description')

    # 1st argument --> numbers ranging from 0 to 9,
    # 2nd argument, row = 2, col = 3
    # Testing - arrayPos = np.array([[332, 43, 62984], [332, 44, 62485], [331, 44, 62442], [330, 44, 62401], [332, 45, 61988], [331, 45, 61945], [330, 45, 61904], [330, 43, 59105], [331, 43, 58868], [145, 305, 26713], [145, 304, 26690], [145, 303, 26669], [146, 305, 26388], [146, 304, 26365], [146, 303, 26344], [147, 305, 26065], [147, 304, 26042], [147, 303, 26021], [310, 293, 20617], [308, 293, 19962], [309, 293, 19700], [442, 285, 16562], [442, 286, 16465], [441, 285, 16325], [441, 286, 16228], [440, 285, 16090], [440, 286, 15993], [246, 134, 14977], [440, 284, 14965], [247, 134, 14810], [246, 133, 14800], [248, 134, 14645], [247, 133, 14633], [246, 132, 14625], [248, 133, 14468], [247, 132, 14458], [248, 132, 14293], [175, 220, 7673], [174, 220, 7618], [173, 220, 7565], [245, 26, 7514], [175, 221, 7508], [245, 27, 7481], [174, 221, 7453], [245, 28, 7450], [151, 391, 7412], [150, 391, 7405], [149, 391, 7400], [173, 221, 7400], [175, 222, 7345], [246, 26, 7345], [246, 27, 7312], [406, 449, 7300], [174, 222, 7290], [246, 28, 7281], [405, 449, 7241], [173, 222, 7237], [404, 449, 7184], [247, 26, 7178], [247, 27, 7145], [247, 28, 7114], [390, 146, 5440], [52, 288, 5045], [52, 289, 4904], [52, 290, 4765], [88, 363, 4201], [88, 362, 4122], [88, 361, 4045], [192, 151, 3205], [436, 401, 3204], [330, 394, 3172], [435, 401, 3145], [329, 394, 3141], [328, 394, 3112], [436, 402, 3109], [434, 401, 3088], [330, 393, 3065], [435, 402, 3050], [329, 393, 3034], [436, 403, 3016], [296, 202, 3005], [328, 393, 3005], [434, 402, 2993], [330, 392, 2960], [435, 403, 2957], [257, 422, 2952], [329, 392, 2929], [296, 203, 2900], [328, 392, 2900], [434, 403, 2900], [258, 422, 2845], [438, 231, 2813], [296, 204, 2797], [376, 367, 2741], [259, 422, 2740], [376, 368, 2692], [376, 369, 2645], [216, 449, 2341], [380, 49, 2320], [380, 48, 2313]])

    # np.random.randint(50, 100,
                #                  size=(450, 2))  # This needs to be the ANMS output (x and y) of the best corners

    r, c = anmsPos.shape  # Size of the ANMS
    #ac print('anms.shape', r, c)
    img_pad = np.pad(image, 50, 'constant',
                     constant_values=0)  # add a border around image for patching, zero to add black countering
    cv2.imwrite('imagepad.png', img_pad)

    for i in range(r):
        patch_center = anmsPos[i]
        #ac print('pos', patch_center)
        patch_x = abs(int(patch_center[0] - patch_size / 2))
        #ac print('x', patch_x)
        patch_y = abs(int(patch_center[1] - patch_size / 2))
        #ac print('y', patch_y) # Here there is a problem with the axis Y for image 2 when the value is less than 20.

        patch = img_pad[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size]
  

        # Apply Gauss blur
        blur_patch = cv2.GaussianBlur(patch, (5, 5), 0)
     
        sub_sample = blur_patch[::5, ::5]
        

        # Re-sahpe to 64x1
        feats = sub_sample.reshape(int((patch_size / 5) ** 2), 1)
        #ac print('Featsub_sampple,shape', feats)
        #ac print('sizeFeats', feats.shape)

        # Make the mean 0
        feats = feats - np.mean(feats)
        # print('feats', feats)

        # Make the variance 1
        feats = feats / (np.std(feats) + epsilon)
       
        feat_desc = np.dstack((feat_desc, feats))
    print('end features descrip')

    return feat_desc[:, :, 1:]


def imageResize(image, w, h):
    #img1 = cv2.imread('../Data/Train/169.jpg')
    #img1 = cv2.imread(image)
    img1_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = int(w)
    height = int(h)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img1_gray, dim, interpolation = cv2.INTER_AREA)
    return resized

def main():

    i = 0 
    #iterateImages() #used to generate arrays and create labesl





def iterateImages():
    #img3 = cv2.imread('/TxtFiles/Train/Set1/3.jpg')
    file1 = open('TxtFiles/DirNamesTrain.txt', 'r')
    Lines = file1.readlines()
    count = 1

    with open('./TxtFiles/LabelsTrain.txt', 'w') as f:
        for line in Lines:


            l = line.strip()
            path = cv2.imread('../Data/' + l + ".jpg")
            groundtruth = supervised(path, count)
            count += 1

            groundtruth= str(groundtruth)[1 : -1]

            f.write(str(groundtruth))


            f.write('\n')

    f.close()

        


def supervised(image, count):
    width, height = 600, 600
    

    img1 = imageResize(image, width, height)

    #cv2.imwrite("4corners.png", img1)


    corners = [[128, 128], [256, 128], [128, 256], [256, 256]]
    '''img1[100][100] = [0,0,255] 
    img1[200][100] = [0,0,255] 
    img1[100][200]= [0,0,255] 
    img1[200][200] = [0,0,255] '''

    #cv2.circle(img1, (100,100) , 10, (0,0,255) , 2) #bottom left
    #cv2.circle(img1, (200,100) , 10, (0,0,255) , 2) #bottom right
    #cv2.circle(img1, (100,200) , 10, (0,0,255) , 2)# top left
    #cv2.circle(img1, (200,200) , 10, (0,0,255) , 2) #top right
    #cv2.imwrite("./alldata/4corners" + str(count) + ".png", img1)


    new_corners, img2= newCorners(corners, img1)



    #cv2.imwrite("newcorners.png", img2)

    groundTruth = extract(corners, new_corners, img2, width, height, count)

    return groundTruth
    #cv2.imwrite("./alldata/warped"+ str(count)+ ".png", warped)


#def supervised():




def extract(corners, new_corners, img, maxWidth, maxHeight, count):
    H = cv2.getPerspectiveTransform(np.float32(corners), np.float32(new_corners)) #Did I do this in the right order? HbA, how do you align?
    H = np.linalg.inv(H) #is this right THIS IS GROUND TRUTH
    out = cv2.warpPerspective(img,H,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

    #widthA = np.sqrt(((new_corners[1][0] - new_corners[0][0]) ** 2) + ((new_corners[1][1] - new_corners[0][1]) ** 2)) 
    #widthB = np.sqrt(((new_corners[3][0] - new_corners[2][0]) ** 2) + ((new_corners[3][1] - new_corners[2][1]) ** 2))

    #heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    #heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))


    #extract from original image patch A with corners

    gt = groundTruth(corners, new_corners)
    patcha  = [[0 for c in range(128)] for r in range(128)]
    patchb = [[0 for c in range(128)] for r in range(128)]
    for i in range(corners[1][0] - corners[0][0]): #x
        for j in range(corners[2][1] - corners[0][1]): #y
            patcha[i][j] = img[i+128][j+128]
  

    patcha = np.array(patcha)






    #cv2.imwrite("./alldata/patcha" +  str(count) + ".png",  patcha)
    for i in range(corners[1][0] - corners[0][0]): #x
        for j in range(corners[2][1] - corners[0][1]): #y
            patchb[i][j] = out[i+128][j+128]

        




    patchb = np.array(patchb)


    together = np.dstack((patcha,patchb))
    #Stacked


    





    
    #cv2.imwrite("./alldata/patchb" + str(count) + ".png", patchb)

    with open("./alldata/test" + str(count) +  ".npy", 'wb') as f:
        np.save(f, together)
    return gt




def groundTruth(corners, new_corners):

  
    blx = new_corners[0][0] - corners[0][0]
    bly = new_corners[0][1] - corners[0][1]
    #bl = [blx, bly]

    brx = new_corners[1][0] - corners[1][0]
    bry = new_corners[1][1] - corners[1][1]
    #br = [brx, bry]

    tlx = new_corners[2][0] - corners[2][0]
    tly = new_corners[2][1] - corners[2][1]
    #tl = [tlx, tly]

    tRx = new_corners[3][0] - corners[3][0]
    tRy = new_corners[3][1] - corners[3][1]
    #tR = [tRx, tRy]

    gradientTruth = [blx, bly, brx, bry, tlx, tly, tRx, tRy]



    return gradientTruth



def newCorners(corners, img):
    newcorners = []

    for coor in corners:

        newx = random.randint(coor[0] - 20, coor[0] + 20)
        newy = random.randint(coor[1] - 20, coor[1] + 20)
        newcorners.append([newx, newy])
        cv2.circle(img, (newx,newy) , 10, (255,0,0) , 2)
  


    return newcorners, img










    #feature1 = featuredescription(img1_gray, patch_size, pts, epsilon, feat_desc)
    #feature2 = featuredescription(img2_gray, patch_size, best2_array, epsilon, feat_desc)


    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    #crop images same size

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
