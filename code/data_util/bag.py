#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:31:50 2019

@author: wuwenjun
"""

import math
import openslide
import numpy as np
from skimage.filters import threshold_multiotsu
import pdb

class Bag:
    """
    Iterator Class for constructing the sliding window Approach

    Designed for image with three channels, image with one channel should add another axis

    """


    def padding(self, img, padding, pad_value=255):
        """
        Padding white borders at different directions of a given patch
        """
        assert img is not None
        h, w, c = img.shape
        pad_top, pad_left, pad_right, pad_bottom = padding
        pad_top, pad_left, pad_right, pad_bottom = int(pad_top), int(pad_left), int(pad_right), int(pad_bottom)
        new_shape = (h+pad_top+pad_bottom, w+pad_left+pad_right, c)
        padded = np.ones(new_shape, dtype=np.uint8) * pad_value

        im_r = pad_top if pad_top > 0 else 0
        im_c = pad_left if pad_left > 0 else 0

        padded[im_r:im_r+h, im_c:im_c+w, :] = img

        return padded

    def __init__(self, im_path=None, slide=None, patch_size=3600, level=0,
        padded=True):
        """
        Initializer for the bag class

        Args:
            im_path: the input image path in openslide supported format

        """
        if im_path is None:
            assert slide is not None
        else:
            assert im_path is not None
            slide = openslide.OpenSlide(im_path)
        self.img = slide
        self.level = level
        self.patch_size = patch_size
        w, h = slide.level_dimensions[level]
        self.shape = (h, w)
        self.w = math.ceil(w / patch_size)
        self.h = math.ceil(h / patch_size)
        self.length = self.h * self.w
        keys = list(range(self.length))
        self.pad_dict = dict(zip(keys, [np.zeros(4, dtype=int) for x in range(len(keys))]))
        self.top_pad = 0
        self.left_pad = 0
        
        """
        padding dictionary:
            [index]: [pad_top, pad_left, pad_right, pad_bottom]
        """
        if padded:
            if w % patch_size != 0:
                for i in range(0, self.length, self.w):
                    self.pad_dict[i][1] = (patch_size - w % patch_size) // 2
                    self.left_pad = (patch_size - w % patch_size) //2
                for i in range(self.w-1, self.length, self.w):
                    self.pad_dict[i][2] = (patch_size - w % patch_size) - self.left_pad
            if h % patch_size != 0:
                for i in range(0, self.w):
                    self.pad_dict[i][0] = (patch_size - h % patch_size) // 2
                    self.top_pad = (patch_size - h % patch_size) // 2

                for i in range(self.length-self.w, self.length):
                    self.pad_dict[i][3] = (patch_size - h % patch_size) - self.top_pad


    def __len__(self):
        """
        Function that return the length of the words/number of
        word in the image

        """
        return self.length


    def bound_box(self, idx):
        """
        Function that return the bounding box of a word given its index
        Args:
            ind: int, ind < number of words

        Returns:
            Bounding box(int[]): [h_low, h_high, w_low, w_high]
        """
        assert idx < self.length, "Index Out of Bound"

# [index]: [pad_top, pad_left, pad_right, pad_bottom]
        left = max((idx % self.w) * (self.patch_size) - self.left_pad, 0)
        right = min(self.shape[1], (idx % self.w) * (self.patch_size) - self.left_pad + self.patch_size)
        top = max(math.floor(idx/self.w) * self.patch_size - self.top_pad, 0)
        bottom = min(self.shape[0], math.floor(idx/self.w) * self.patch_size - self.top_pad + self.patch_size)

        return [top, bottom, left, right]

    def __getitem__(self, idx):
        """
        Function that returns the word at a index
        Args:
            idx: int, ind < number of words

        """
        if idx >= self.length:
            raise StopIteration

        top, bottom, left, right = self.bound_box(idx)
        crop = self.img.read_region((left, top), self.level, (right-left, bottom-top)).convert('RGB')
        crop = np.array(crop)
        crop = self.padding(crop, self.pad_dict[idx])
        return crop


    def calculate_label_from_roi_bbox(self, roi_bbox):

        """
        Get small image for thresholding tissue
        """
        if self.img.level_count == 1:
            w, h = self.img.level_dimensions[0]
            new_w, new_h = (w//4, h//4)
            slide_small = self.img.get_thumbnail((new_w, new_h))
        else:
            level = self.img.get_best_level_for_downsample(16)
            slide_small = self.img.read_region((0, 0), level,
                                                 self.img.level_dimensions[level]).convert('RGB')
            slide_small = np.array(slide_small)

        threshold = get_threshold_otsu(slide_small)

        pos_ind = set()
        h, w = self.shape
        # Bounding box(int[]): [h_low, h_high, w_low, w_high]
        for h_low, h_high, w_low, w_high in roi_bbox:
            h_low  = min(h-1, h_low + self.top_pad)
            h_high = min(h-1, h_high + self.top_pad)
            w_low = min(w-1, w_low + self.left_pad)
            w_high = min(w-1, w_high + self.left_pad)
            #if h_high >= h and w_high >= w:
            if h_high > h or w_high > w:
                print("Size incompatible for case")
                print("Bounding box: {}, {}, {}, {}".format(h_low, h_high, w_low, w_high))
                print("WSI size: {}, {}". format(h, w))
                h_high = min(h, h_high)
                w_high = min(w, w_high)
            ind_w_low = max(int(w_low // (self.patch_size)), 0)
            ind_w_high = int(min(max(w_high // self.patch_size, 0), self.w - 1))
            ind_h_low = int(max(h_low//self.patch_size, 0))
            ind_h_high = int(min(max(h_high//self.patch_size, 0), self.h - 1))
            for i in range(ind_h_low, ind_h_high + 1):
                pos_ind.update(range(i * self.w + ind_w_low,
                   i * self.w + ind_w_high + 1))
        pos_ind_copy = list(pos_ind)
        # print(pos_ind_copy)

        for ind in pos_ind:
            h_low, h_high, w_low, w_high = self.bound_box(ind)
            h_low -= self.top_pad
            h_high -= self.top_pad
            w_low -= self.left_pad
            w_high -= self.left_pad
            if (not self.checkROI([h_low, h_high, w_low, w_high],
                                  roi_bbox)) or (checkEmptyImage(self.__getitem__(ind), threshold)):

                pos_ind_copy.remove(ind)
        pos_ind = np.sort(list(pos_ind_copy))
        # print(pos_ind)
        return pos_ind


    def checkROI(self, idx_bbox, bboxes):
        [row_up, row_down, col_left, col_right] = idx_bbox
        check_bbox = {'x1': col_left, 'y1': row_up,
                      'x2': col_right, 'y2': row_down}
        check_area = 0
        for bb in bboxes:
            [row_up, row_down, col_left, col_right] = bb

            roi_bbox = {'x1': col_left, 'y1': row_up,
                      'x2': col_right, 'y2': row_down}
            intersection_area, perc = get_iou(roi_bbox, check_bbox)
            check_area += intersection_area
        # print('area: {}'.format(check_area))
        if check_area / (self.patch_size ** 2) < 0.1:
            return False
        return True

def checkEmptyImage(image, thresholds):
    regions = apply_threshold_otsu(image, thresholds)
    val, count = np.unique(regions, return_counts=True)
    if len(val) == 1:
        return True
    else:
        ind = np.where(val==1)[0]
        if ind:
            tissue_count = count[ind[0]]
            if tissue_count / (image.shape[0] * image.shape[1]) > 0.95:
                return True
            else:
                return False
    return True


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0, 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return intersection_area, iou


def get_threshold_otsu(image):
    # return labels [0, 1]
    gray = rgb_to_gray(image)
    thresholds = threshold_multiotsu(gray, classes=2)
    return thresholds


def rgb_to_gray(image):
    image = np.array(image)
    assert len(image.shape) > 2
    rgb_weights = [0.5870, 0.1140, 0.2989]
    gray = np.dot(image[..., :3], rgb_weights)
    return gray


def apply_threshold_otsu(image, thresholds):
    gray = rgb_to_gray(image)
    regions = np.digitize(gray, bins=thresholds)
    return regions

