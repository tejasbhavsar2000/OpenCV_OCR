#!/usr/bin/env python2.7
# coding=utf8

import numpy as np
import cv2
import cv2.cv as cv

import os
import copy

from opencv_ocr import opencv_ocr

if __name__ == '__main__' and __package__ is None:
    _opencv_ocr = opencv_ocr()
    # Using the SVM algorithm
    _svm_model = _opencv_ocr.svm_init('digits_svm.dat')
    # Read all the pictures from the img_data file.
    work_path = '%s/img_data/data4/' % os.getcwd()
    images_path = os.listdir(work_path)
    # Specify a single test.
    # images_path = ['10.jpg']
    # Then cycle through the list of original ones for recognition
    for index, _path in enumerate(images_path):
        # read from the picture
        if _path[_path.rfind('.'):] != '.jpg':
            continue
        image = cv2.imread(work_path + _path)
        if image.shape[0] > image.shape[1]:
            image = np.rot90(image, 1)
        # Precisely position the string, also accounts for skewed images, (skew correction)
        img_binary = _opencv_ocr._character_location(image)
        # Split EVERY Character
        img_binary_copy = img_binary.copy()
        contours, hierarchy = cv2.findContours(
            img_binary_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Get the coordinates for each and every character
        _box_shape = []
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            _box_shape.append([x, x+w, y, y+h])
        _box_shape = sorted(_box_shape, key=lambda _box: _box[0])

        # List all the possible characters
        _image_list = [img_binary[_box[2]:_box[3], _box[0]:_box[1]]
                       for _box in _box_shape]
        # Dynamic size
        _image_h_list = [_box[3]-_box[2] for _box in _box_shape]
        _median_h = int(np.median(_image_h_list))
        if _median_h < 6:
            _median_h = 6
        # Filter out some small images that are not characters,
        # split some connected characters which might be still undivided
        _opencv_ocr._correct_char_image(
            _image_list, (_median_h-5, _median_h+5))
        # Throw the correct character picture list into the SVM model for recognition
        _string = _opencv_ocr._svm_classify_string(_svm_model, _image_list)
        print 'classify %s is :' % _path, _string

        # Output every single character picture for training
        for __idx, (__img, _ch) in enumerate(zip(_image_list, _string)):
            cv2.imwrite('%s/char_good/ch%d_%d_%s.jpg' %
                        (os.getcwd(), index, __idx, _ch), __img)

        # Output the binarized image of the original image
        # if len(_string) != 0:
        #     for _idx, __img in enumerate(_image_list):
        #         cv2.imshow('img%d'%_idx, __img)
        #     cv2.imshow('imagefull', img_binary)
        #     cv2.waitKey()
