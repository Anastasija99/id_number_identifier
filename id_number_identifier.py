# Digital Image Processing Project

import glob
import operator

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
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
        if len(images) == 1:
            cv2.imshow(title, image)
        else:
            cv2.imshow(title + ' ' + str(image_index + 1), image)
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
    for ind, image in enumerate(images_оpened):
        image_cropped = cv2.resize(image[310:340, 5:160], (400, 250))
        plt.suptitle('Extracted elements from ID ' + str(ind + 1), fontsize=14)
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
            if index + 2 >= 9:
                break

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


# Method for evaluating the matched elements
def evaluate_elements(matched_elements):
    id_number = ''
    for value in matched_elements.values():
        if value == 10:
            id_number = ''.join([id_number, 'A'])
        elif value == 11:
            id_number = ''.join([id_number, 'T'])
        else:
            id_number = ''.join([id_number, str(value)])

    return id_number


# Method for matching the elements from the ID image to the pattern
def match_elements_to_pattern(images_elements, pattern_elements):
    result_id_number = dict()
    for ID_index, image in enumerate(images_elements):
        matched_elements = dict()
        for image_element_index, image_element in enumerate(image):
            element_score = dict()
            for pattern_index, pattern_element in enumerate(pattern_elements):
                matching_result = cv2.matchTemplate(image_element, pattern_element, cv2.TM_CCOEFF)
                score = cv2.minMaxLoc(matching_result)[1]
                element_score[pattern_index] = score
            element_score = dict(sorted(element_score.items(), key=operator.itemgetter(1), reverse=True))
            matched_elements[image_element_index] = list(element_score.keys())[0]
        id_number = evaluate_elements(matched_elements)
        result_id_number['ID ' + str(ID_index + 1)] = id_number
        print(f'ID {ID_index + 1} : {id_number}')
    return result_id_number


# Method for showing images with the id number written on them
def show_images_with_id_number(images_color, matching_result):
    for index, image in enumerate(images_color):
        cv2.rectangle(image, (170, 345), (15, 300), (0, 255, 255), 2)
        cv2.putText(image, matching_result['ID ' + str(index + 1)], (25, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 255), 2)
        cv2.imshow('ID ' + str(index + 1), image)
        cv2.waitKey(0)


# Method for detecting edges of the id card and cropping the images
def detect_edges(images_color, images_gray):
    cropped_images_color = list()
    cropped_images_gray = list()
    for image, image_gray in zip(images_color, images_gray):
        blurred = cv2.blur(image, (3, 3))
        canny = cv2.Canny(blurred, 100, 255)
        pts = np.argwhere(canny > 0)
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        tagged = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
        # cv2.imshow("tagged", tagged)
        # cv2.waitKey()
        cropped = image[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (600, 350))
        cropped_images_color.append(cropped)
        cropped_gray = image_gray[y1:y2, x1:x2]
        cropped_gray = cv2.resize(cropped_gray, (600, 350))
        cropped_images_gray.append(cropped_gray)

    return cropped_images_color, cropped_images_gray


if __name__ == '__main__':
    # Glob for the images of the id cards
    image_database = glob.glob('./images/*.jpg')

    # Glob for the image of the pattern
    pattern = glob.glob('./pattern/*.jpg')[0]

    # Reading the images from the database in color
    images_color = read_images_color(image_database)

    # Reading the images from the database in grayscale
    images_gray = read_images_gray(image_database)

    # Calling method for detecting edges
    images_color, images_gray = detect_edges(images_color, images_gray)

    # Reading the pattern in grayscale
    pattern_gray = cv2.imread(pattern, cv2.IMREAD_GRAYSCALE)
    # Double thresholding for better results
    pattern_gray = cv2.threshold(pattern_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
    pattern_gray = cv2.threshold(pattern_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    # Calling method for showing ID images in color
    show_images(images_color, 'ID Image')
    # Calling method for showing ID images in gray
    # show_images(images_gray, 'ID Image gray')
    # Calling method for showing pattern image
    show_images([pattern_gray], 'Pattern Image')

    # Calling method open_images on ID images
    images_оpened = open_images(images_gray)

    # Calling method extract_pattern_elements on the pattern
    pattern_elements = extract_pattern_elements(pattern_gray)
    # Calling method extract_elements on ID images
    images_elements = extract_elements(images_оpened)

    # Calling method match_elements_to_pattern
    matching_result = match_elements_to_pattern(images_elements, pattern_elements)

    show_images_with_id_number(images_color, matching_result)
