# Digital Image Processing Project
import glob
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils import contours
import numpy as np
import os

# Method for reading the images in color
def read_images_color(images_glob):
    images_list = list()
    for image in images_glob:
        image_new = cv2.imread(image)
        images_list.append(image_new)

    return images_list


# Method for reading the images in grayscale
def read_images_gray(images_glob):
    images_list = list()
    for image_g in images_glob:
        image_new = cv2.imread(image_g, cv2.IMREAD_GRAYSCALE)
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

# Method for showing the created elements of the pattern
def show_created_pattern_images(pattern_images_opened):
    pattern_list = []
    for ind, pattern_image in enumerate(pattern_images_opened):
        contour = get_contours(pattern_image)
        #for index, c in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(contour[0])
        element = pattern_image[y:y + h, x:x + w]
        element = cv2.resize(element, (200, 250))
        pattern_list.append(element)
        plt.subplot(1, 1, 1)
        plt.title('image - ' + str(ind))
        plt.imshow(element, cmap='gray')
        plt.show()
    return pattern_list

# Method for making the pattern and saving it in the directory pattern
def combine_pattern_images(pattern_images):
    black = cv2.imread("pattern/space/black_photo.jpg", cv2.IMREAD_GRAYSCALE)
    black = cv2.resize(black,(50,250))

    p = black
    for index,i in enumerate(pattern_images):
        p = np.concatenate((p,i), axis=1)
        p = np.concatenate((p, black), axis=1)
        cv2.imwrite(os.path.join("./pattern/black_and_white", str(index) + '.jpg'), i)


    plt.imshow(p,cmap='gray')
    plt.show()
    cv2.imwrite(os.path.join("./pattern" , 'pattern.jpg'), p)

if __name__ == "__main__":
    # Glob for the images for the pattern
    pattern_database = glob.glob("./pattern/pattern_images/*jpg")

    # Reading the images from the database in color
    pattern_images_color = read_images_color(pattern_database)

    # Reading the images from the database in grayscale
    pattern_images_gray = read_images_gray(pattern_database)

    # Showing the images
    #show_images(pattern_images_color)
    #show_images(pattern_images_gray)

    # Calling the method open_images
    pattern_images_opened = open_images(pattern_images_gray)
    # Calling the method show_created_pattern_images
    pattern_list = show_created_pattern_images(pattern_images_opened)

    # Calling the method combine_pattern_images
    combine_pattern_images(pattern_list)

