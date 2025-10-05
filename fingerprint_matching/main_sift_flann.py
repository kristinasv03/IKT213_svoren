import cv2

image1 = cv2.imread('UiA front1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('UiA front3.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Draw keypoints on the image
#output_image = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 0))

FLANN_INDEX_KDTREE = 1  # Algorithm type for SIFT/SURF
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # K-D Tree with 5 trees
search_params = dict(checks=50)  # Number of times the tree is recursively traversed

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Perform matching
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('UiA_sift_flann.jpg', match_img)
cv2.imshow('Sift Keypoints', match_img)
cv2.waitKey(0)