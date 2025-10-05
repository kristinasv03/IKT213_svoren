import cv2

image1 = cv2.imread('UiA front1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('UiA front3.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)

#Sort matches based on distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('UiA_orb_bf.jpg', match_img)
cv2.imshow('ORB Keypoints', match_img)
cv2.waitKey(0)