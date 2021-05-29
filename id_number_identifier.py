# Digital Image Processing Project

import glob

import cv2
import imutils
import matplotlib.pyplot as plt
from imutils import contours


# Method for reading the images in color and resizing
def read_images_color(images_glob):
    images_list = list()
    for image in images_glob:
        image_new = cv2.imread(image)
        image_new = cv2.resize(image_new, (600, 350))
        images_list.append(image_new)

    return images_list


# Method for reading the images in grayscale and resizing
def read_images_gray(images_glob):
    images_list = list()
    for image_g in images_glob:
        image_new = cv2.imread(image_g, cv2.IMREAD_GRAYSCALE)
        image_new = cv2.resize(image_new, (600, 350))
        images_list.append(image_new)

    return images_list


# Method for showing the images
def show_images(images):
    for image_index, image in enumerate(images):
        cv2.imshow("ID Image", image)
        cv2.waitKey(0)


# Method for thresholding and using morphological operation Opening
def open_images(images_set):
    opened_images_list = list()
    for img in images_set:
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, elem)
        opened_images_list.append(opened)

    return opened_images_list


# Method for finding contours on the cropped image
def get_contours(image_cropped):
    contour = cv2.findContours(image_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = contours.sort_contours(contour, method="left-to-right")[0]
    return contour


# Method for TO DO:
def extract_numbers(images_оpened):
    for image in images_оpened:
        image_cropped = cv2.resize(image[300:345, 5:160], (400, 250))
        plt.suptitle("Extracted elements", fontsize=14)
        plt.subplot(3, 3, 1)
        plt.imshow(image_cropped, cmap='gray')

        contour = get_contours(image_cropped)
        for index, c in enumerate(contour):
            (x, y, w, h) = cv2.boundingRect(c)
            element = image_cropped[y:y + h, x:x + w]
            element = cv2.resize(element, (200, 250))
            plt.subplot(3, 3, 2 + index)
            plt.imshow(element, cmap='gray')
        plt.show()
    # ...
    # return string "A00000"


if __name__ == "__main__":
    # Glob for the images of the id cards
    image_database = glob.glob("./images/*.jpg")

    # Glob for the image of the pattern
    pattern = glob.glob("./pattern/*.png")[0]

    # Reading the images from the database in color
    images_color = read_images_color(image_database)

    # Reading the images from the database in grayscale
    images_gray = read_images_gray(image_database)

    # Reading the pattern in grayscale
    pattern_gray = cv2.imread(pattern, cv2.IMREAD_GRAYSCALE)

    # Tresholding and inverting the pattern image
    pattern_gray = cv2.threshold(pattern_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    # Calling method for showing images in color
    show_images(images_color)
    # Calling method for showing images in gray
    show_images(images_gray)

    # Showing the pattern in gray
    cv2.imshow("Pattern image", pattern_gray)
    cv2.waitKey(0)

    # Calling method opened_images on grayscale images
    images_оpened = open_images(images_gray)

    # TO DO:
    numbers_sting = extract_numbers(images_оpened)
