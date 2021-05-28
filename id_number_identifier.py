#Digital Image Processing Project

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

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
        image_new_g = cv2.imread(image_g,cv2.IMREAD_GRAYSCALE)
        images_list.append(image_new_g)

    return images_list

# Method for showing the images
def show_images(images):

    for image_index, image in enumerate(images):
        cv2.imshow("title", image)
        cv2.waitKey(0)



if __name__ == "__main__":
    print("initial")

    # Glob for the images of the id cards
    image_database = glob.glob("./images/*.jpg")

    # Glob for the image of the pattern
    pattern = glob.glob("./pattern/*.png")



    # Reading the images from the database in color
    images_color = read_images_color(image_database)

    # Reading the images from the database in grayscale
    images_gray = read_images_gray(image_database)

    # Reading the pattern in grayscale
    pattern_gray = read_images_gray(pattern)[0]


    # Tresholding and inverting the pattern image
    pattern_gray = cv2.threshold(pattern_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    # Calling method for showing images in color
    show_images(images_color)
    # Calling method for showing images in gray
    show_images(images_gray)

    # Showing the pattern in gray
    cv2.imshow("title", pattern_gray)
    cv2.waitKey(0)