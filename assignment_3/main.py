import cv2
import numpy as np

image = cv2.imread('lambo.png')
shapes = cv2.imread('shapes-1.png')
template = cv2.imread('shapes_template.jpg')
w, h = template.shape[:-1]

def sobel_edge_detection(image):
    img_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    return sobelxy

def canny_edge_detection(image, threshold1=50, threshold2=50):
    img_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    canny = cv2.Canny(img_blur, threshold1, threshold2)
    return canny

def template_match(shapes, template):
    shapes_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    match = cv2.matchTemplate(shapes_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(match >= threshold)
    result = shapes.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return result

def resize(image, scale_factor: int , up_or_down: str):
    result_image = image.copy()
    if up_or_down.lower() == "up":
        for _ in range(int(np.log2(scale_factor))):
            rows, cols = result_image.shape[:2]
            result_image = cv2.pyrUp(result_image, dstsize=(2 * cols, 2 * rows))
    elif up_or_down.lower() == "down":
        for _ in range(int(np.log2(scale_factor))):
            rows, cols = result_image.shape[:2]
            result_image = cv2.pyrDown(result_image, dstsize=(cols // 2, rows // 2))
    return result_image

def main():
    sobel_image = sobel_edge_detection(image)
    cv2.imwrite('sobel_lambo.png', sobel_image)
    #cv2.imshow('sobel_lambo.png', sobel_image)
    #cv2.waitKey(0)

    canny_image = canny_edge_detection(image)
    cv2.imwrite('canny_lambo.png', canny_image)
    #cv2.imshow('canny_lambo.png', canny_image)
    #cv2.waitKey(0)

    template_match_image = template_match(shapes, template)
    cv2.imwrite('template_match.png', template_match_image)
    #cv2.imshow('template_match.png', template_match_image)
    #cv2.waitKey(0)

    resize_down_image = resize(image, scale_factor=2, up_or_down="down")
    cv2.imwrite('resized_down_lambo.png', resize_down_image)
    #cv2.imshow('resize_down_lambo.png', resize_down_image)


    resize_up_image = resize(image, scale_factor=2, up_or_down="up")
    cv2.imwrite('resized_up_lambo.png', resize_up_image)
    #cv2.imshow('resize_up_lambo.png', resize_up_image)
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()