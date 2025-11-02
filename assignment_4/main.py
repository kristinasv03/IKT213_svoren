import cv2
import numpy as np

reference = cv2.imread('reference_img.png')
align_this = cv2.imread('align_this.jpg')

MIN_MATCH_COUNT = 4

def reference_image(image):
    out = image.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    out[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite('harris.png', out)
    return out

def SIFT(image_to_align, reference_image, max_features, good_match_precent):
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_aln = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
    kp_aln, des_aln = sift.detectAndCompute(gray_aln, None)

    if des_ref is None or des_aln is None:
        print("No descriptors found.")
        return None, None

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(des_ref, des_aln, k=2)

    ratio_good = []
    for m, n in knn_matches:
        if m.distance < good_match_precent * n.distance:
            ratio_good.append(m)

    if len(ratio_good) == 0:
        print("No good matches after ratio test.")
        return None, None

    ratio_good = sorted(ratio_good, key=lambda m: m.distance)

    limited_matches = ratio_good[:max_features]

    if len(limited_matches) < MIN_MATCH_COUNT:
        print(f"Not enough matches are found - {len(limited_matches)}/{MIN_MATCH_COUNT}")

        matches_img = cv2.drawMatches(
            reference_image, kp_ref,
            image_to_align, kp_aln,
            limited_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return None, matches_img

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in limited_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_aln[m.trainIdx].pt for m in limited_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography failed.")
        matches_img = cv2.drawMatches(
            reference_image, kp_ref,
            image_to_align, kp_aln,
            limited_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return None, matches_img

    H_inv = np.linalg.inv(H)

    h_ref, w_ref = gray_ref.shape[:2]

    aligned = cv2.warpPerspective(
        image_to_align,
        H_inv,
        (w_ref, h_ref),
        flags=cv2.INTER_LINEAR
    )

    matchesMask = mask.ravel().tolist()

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    img_matches = cv2.drawMatches(
        reference_image, kp_ref,
        image_to_align, kp_aln,
        limited_matches,
        None,
        **draw_params
    )

    return aligned, img_matches



def main():
    _ = reference_image(reference)
    #cv2.imshow('reference_corners.png', ref_image)

    aligned, matches = SIFT(align_this, reference, max_features=10, good_match_precent=0.7)
    if matches is not None:
        cv2.imwrite('matches.png', matches)
    else:
        print("No matches visualization available.")

    if aligned is not None:
        cv2.imwrite('aligned.png', aligned)
    else:
        print("No aligned image was produced.")

if __name__ == '__main__':
    main()