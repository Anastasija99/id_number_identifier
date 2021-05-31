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
def show_images(images, title):
    for image_index, image in enumerate(images):
        image = cv2.resize(image, (600, 350))
        cv2.imshow(title, image)
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
    contour = contours.sort_contours(contour, method='left-to-right')[0]
    return contour


# Method for extracting separate elements from the ID images
def extract_elements(images_оpened):
    images_extracted_elements = list()
    for image in images_оpened:
        image_cropped = cv2.resize(image[300:345, 5:160], (400, 250))
        plt.suptitle('Extracted elements', fontsize=14)
        plt.subplot(3, 3, 1)
        plt.imshow(image_cropped, cmap='gray')
        extracted_elements = list()
        contours = get_contours(image_cropped)
        for index, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            element = image_cropped[y:y + h, x:x + w]
            element = cv2.resize(element, (200, 250))
            extracted_elements.append(element)
            plt.subplot(3, 3, 2 + index)
            plt.imshow(element, cmap='gray')

        plt.show()
        images_extracted_elements.append(extracted_elements)

    return images_extracted_elements


# Method for extracting separate elements from the pattern image
def extract_pattern_elements(pattern):
    extracted_pattern_elements = list()
    contours = get_contours(pattern)
    contours = [contour for index, contour in enumerate(contours) if index != 2]
    plt.suptitle('Pattern elements', fontsize=14)
    for index, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        element = pattern[y:y + h, x:x + w]
        element = cv2.resize(element, (200, 250))
        extracted_pattern_elements.append(element)
        plt.subplot(3, 4, index + 1)
        plt.imshow(element, cmap='gray')
        if index == 11:
            break
    plt.show()
    return extracted_pattern_elements


if __name__ == '__main__':
    # Glob for the images of the id cards
    image_database = glob.glob('./images/*.jpg')

    # Glob for the image of the pattern
    pattern = glob.glob('./pattern/*.jpg')[0]

    # Reading the images from the database in color
    images_color = read_images_color(image_database)

    # Reading the images from the database in grayscale
    images_gray = read_images_gray(image_database)

    # Reading the pattern in grayscale
    pattern_gray = cv2.imread(pattern, cv2.IMREAD_GRAYSCALE)
    # Double thresholding for better results
    pattern_gray = cv2.threshold(pattern_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
    pattern_gray = cv2.threshold(pattern_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    # Calling method for showing ID images in color
    show_images(images_color, 'ID Image')
    # Calling method for showing ID images in gray
    show_images(images_gray, 'ID Image gray')
    # Calling method for showing pattern image
    show_images([pattern_gray], 'Pattern Image')

    # Calling method open_images on ID images
    images_оpened = open_images(images_gray)

    # Calling method extract_pattern_elements on the pattern
    pattern_elements = extract_pattern_elements(pattern_gray)
    # Calling method extract_elements on ID images
    images_elements = extract_elements(images_оpened)

