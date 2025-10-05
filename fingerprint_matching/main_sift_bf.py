import cv2
import numpy as np

image1 = cv2.imread('UiA front1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('UiA front3.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Draw keypoints on the image
#output_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('UiA_sift_bf.jpg', match_img)
cv2.imshow('Sift Keypoints', match_img)
cv2.waitKey(0)