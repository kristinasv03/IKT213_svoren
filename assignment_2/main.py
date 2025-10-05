import cv2
import numpy as np

image = cv2.imread('lena-2.png')

border_width = 100
x_0 = 80
x_1 = image.shape[1] - 130
y_0 = 80
y_1 = image.shape[0] - 130
width = 200
height = 200
h, w, c = image.shape
emptyPictureArray = np.zeros((h, w, 3), dtype=np.uint8)

def padding (image, border_width):
    return cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)

def crop (image, x_0, x_1, y_0, y_1):
    return image[y_0:y_1, x_0:x_1]

def resize (image, width, height):
    return cv2.resize(image, (width, height))

def copy (image, emptyPictureArray):
    emptyPictureArray[0:h, 0:w] = image
    return emptyPictureArray

def grayscale (image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv (image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted(image, emptyPictureArray, hue):
    height, width, _ = image.shape
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            new_r = (int(r) + hue) % 256
            new_g = (int(g) + hue) % 256
            new_b = (int(b) + hue) % 256
            emptyPictureArray[i, j] = [new_r, new_g, new_b]
    return emptyPictureArray

def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

def main():
    padded_image = padding(image, border_width)
    cv2.imwrite('padded_lena.png', padded_image)

    cropped_image = crop(image, x_0, x_1, y_0, y_1)
    cv2.imwrite('cropped_lena.png', cropped_image)

    resized_image = resize(image, width, height)
    cv2.imwrite('resized_lena.png', resized_image)

    emptyPictureArray = np.zeros((h, w, 3), dtype=np.uint8)
    copied_image = copy(image, emptyPictureArray)
    cv2.imwrite('copied_lena.png', copied_image)

    grayscale_image = grayscale(image)
    cv2.imwrite('grayscale_lena.png', grayscale_image)

    hsv_image = hsv(image)
    cv2.imwrite('hsv_lena.png', hsv_image)

    emptyPictureArray = np.zeros((h, w, 3), dtype=np.uint8)
    hue_shifted_image = hue_shifted(image, emptyPictureArray,50)
    cv2.imwrite('hue_shifted_lena.png', hue_shifted_image)

    smoothing_image = smoothing(image)
    cv2.imwrite('smoothing_lena.png', smoothing_image)

    rotation_90_image = rotation(image, 90)
    cv2.imwrite('rotation_90_lena.png', rotation_90_image)
    rotation_180_image = rotation(image, 180)
    cv2.imwrite('rotation_180_lena.png', rotation_180_image)

    #cv2.imshow('padded_lena.png', padded_image)
    #cv2.imshow('cropped_lena.png', cropped_image)
    #cv2.imshow('resized_lena.png', resized_image)
    #cv2.imshow('copied_lena.png', copied_image)
    #cv2.imshow('grayscale_lena.png', grayscale_image)
    #cv2.imshow('hsv_lena.png', hsv_image)
    #cv2.imshow('hue_shifted_lena.png', hue_shifted_image)
    #cv2.imshow('smoothing_lena.png', smoothing_image)
    #cv2.imshow('rotation_90_lena.png', rotation_90_image)
    #cv2.imshow('rotation_180_lena.png', rotation_180_image)
    #cv2.waitKey(0)

if __name__ == "__main__":
    main()
