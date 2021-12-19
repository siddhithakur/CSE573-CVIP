"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import math
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random
def get_keypoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp,des


def getMatches(image1, kpl, image2, kpr):
    eta = 0.75

    matches = []

    first_two = []

    for i in range(len(image1)):
        first_two = []
        for j in range(len(image2)):
            distance = math.sqrt(np.sum((np.array(image1[i]) - np.array(image2[j])) ** 2))
            first_two.append([i, j, distance])
        first_two = sorted(first_two, key=lambda x: x[2])
        first_two = first_two[:2]
        matches.append([first_two[0][0], first_two[0][1], first_two[1][1], first_two[0][2], first_two[1][2]])
    good_matches = []
    for i in matches:
        if i[3] < eta * i[4]:
            good_matches.append(i)
    good_matches = sorted(good_matches, key=lambda x: x[3])
    print(len(good_matches))
    keypoints1 = []
    keypoints2 = []

    for i in good_matches:
        keypoints1.append(kpl[i[0]])
        keypoints2.append(kpr[i[1]])

    return keypoints1, keypoints2


def ransac(matches, t, k):
    largest_inliers = []
    for i in range(k):

        random_points = matches[np.random.choice(len(matches), size=4, replace=False)]
        m = []
        for points in random_points:
            m.append([points.item(0), points.item(1), 1, 0, 0, 0, -points.item(2) * points.item(0),
                      -points.item(2) * points.item(1), -points.item(2)])
            m.append([0, 0, 0, points.item(0), points.item(1), 1, -points.item(3) * points.item(0),
                      -points.item(3) * points.item(1), -points.item(3)])
        u, s, v = np.linalg.svd(m)
        h = np.reshape(v[-1], (3, 3))
        h = h / h.item(8)
        inliers = []

        for i in range(len(matches)):

            p1 = []
            p1.append([matches[i].item(0), matches[i].item(1), 1])
            normp1 = (np.dot(h, np.array(p1).T)) / np.dot(h, np.array(p1).T)[2, :]
            p2 = []
            p2.append([matches[i].item(2), matches[i].item(3), 1])
            residual = np.array(p2).T - normp1
            residual = np.linalg.norm(residual)
            if residual < t:
                inliers.append(matches[i])

        if len(inliers) > len(largest_inliers):
            largest_inliers = inliers
            homography = h


    return homography
def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    kpl, desl = get_keypoints(left_img)
    kpr, desr = get_keypoints(right_img)

    matches = []

    keypoints1, keypoints2 = getMatches(desl, kpl, desr, kpr)
    for kp in range(len(keypoints1)):
        l1 = list(keypoints1[kp].pt)
        l1.extend(list(keypoints2[kp].pt))
        matches.append(l1)
    print(matches)
    matches = np.matrix(matches)
    t = 4.5
    kv = 1000

    homography =ransac(matches, t,kv)

    width_1, height_1 = right_img.shape[0], right_img.shape[1]
    width_2, height_2 = left_img.shape[0], left_img.shape[1]

    r = np.float32([[0, 0], [0, width_1], [height_1, width_1], [height_1, 0]]).reshape(-1, 1, 2)
    l = np.float32([[0, 0], [0, width_2], [height_2, width_2], [height_2, 0]]).reshape(-1, 1, 2)

    l = cv2.perspectiveTransform(l, homography)
    print(l)
    finaldimensions = np.concatenate((l, r), axis=0)
    x = []
    y = []
    print(finaldimensions)
    for i in finaldimensions:
        x.append(i[0][0])
        y.append(i[0][1])

    xmin = int(min(x))
    ymin = int(min(y))
    xmax = int(max(x))
    ymax = int(max(y))

    affinity_matrix = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    stitch = cv2.warpPerspective(left_img, affinity_matrix.dot(homography),
                                 (xmax - xmin, ymax - ymin))

    stitch[-ymin:-ymin + width_2, -xmin:-xmin + height_2] = right_img
    result_img = stitch
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


