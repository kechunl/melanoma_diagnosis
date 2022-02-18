import numpy as np
import os
import cv2
import sys
import glob
import pdb
import argparse
import subprocess
from bag import Bag
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000


def stitch(args):
	big_image_list = glob.glob(os.path.join(args.original_dir, '**', '*.tif'))
	# big_image_list = [p for p in big_image_list if 'Sox' not in p]

	count = 0
	for big_image_path in big_image_list:
		print('Stitching ', big_image_path)
		# try:
		assert os.path.exists(big_image_path)
		# assert os.path.exists(big_image_path.replace('_big', '_Sox_big'))
		
		if 'big' not in big_image_path:
		# convert tiff to bigtiff for temporary use
			subprocess.call("vips tiffsave {} {} --bigtiff --tile --compression lzw".format(big_image_path, big_image_path.split('.')[0]+'_big.tif'), shell=True)
		
		big_image_path = big_image_path.split('.')[0]+'_big.tif'

		# HE_bag = Bag(im_path=big_image_path, patch_size=512, level=0) # 20x
		# h, w = int(HE_bag.img.level_dimensions[0][1]/2), int(HE_bag.img.level_dimensions[0][0]/2)
		# full_image = np.zeros((h, w), dtype=np.uint8)

		HE_bag = Bag(im_path=big_image_path, patch_size=256, level=0) # 10x
		h, w = int(HE_bag.img.level_dimensions[0][1]), int(HE_bag.img.level_dimensions[0][0])
		full_image = np.zeros((h, w), dtype=np.uint8)

		slice_name = os.path.basename(big_image_path).replace('_big.tif', '')
		# patch_paths = glob.glob(os.path.join(args.patch_dir, '*_mask', '{}_*.png'.format(slice_name)))
		patch_paths = glob.glob(os.path.join(args.patch_dir, '{}_*.png'.format(slice_name)))
		assert len(patch_paths) > 0

		for patch_path in patch_paths:
			patch_ind = int(os.path.basename(patch_path).split('.')[0].split('_')[-1])
			# position in full mask
			bbox = HE_bag.bound_box(patch_ind)

			# top, bottom, left, right = [int(p/2) for p in bbox]
			top, bottom, left, right = bbox

			# get padding
			pad_dict = HE_bag.pad_dict[patch_ind] # [pad_top, pad_left, pad_right, pad_bottom]
			# pad_top, pad_left, pad_right, pad_bottom = [int(p/2) for p in pad_dict]
			pad_top, pad_left, pad_right, pad_bottom = pad_dict

			# patch image
			patch = np.array(Image.open(patch_path))
			psize = patch.shape[0]
			unpadded_patch = patch[pad_top:pad_top+bottom-top, pad_left:pad_left+right-left]
			full_image[top:bottom, left:right] = unpadded_patch

		os.makedirs(os.path.join(args.save_dir, slice_name.split('_')[0]), exist_ok=True)
		Image.fromarray(full_image).save(os.path.join(args.save_dir, slice_name.split('_')[0], '{}.tif'.format(slice_name)))
		count += 1

		if 'big' in big_image_path:
			os.remove(big_image_path)
		# except:
		# 	print('Error in ', big_image_path)
	print('Finish Stitching {} images'.format(count))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--patch_dir', type=str, help='path to patches', default=None)
	parser.add_argument('--save_dir', type=str, help='path to save the stitched images.', default=None)
	parser.add_argument('--original_dir', type=str, help='path to original images (bigtiff preferred)')
	args = parser.parse_args()

	# args.original_dir = '/projects/patho2/melanoma/melanocyte/images/'
	os.makedirs(args.save_dir, exist_ok=True)

	stitch(args)

	# python stitch_patch.py --patch_dir /projects/patho4/Kechun/melanocyte_detection/melanocyte_data/datasets/melanocyte_10x_256_new --save_dir /projects/patho4/Kechun/diagnosis/dataset/melanocyte_mask