"""
Extract melanocyte cell graphs
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
import h5py
import torch
from dgl.data.utils import save_graphs
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt

from skimage.measure import label
from skimage.measure import regionprops

from model.melanocyte_feature_extraction import DeepFeatureExtractor, HandcraftedFeatureExtractor, GANFeatureExtractor
from histocartography.preprocessing import (
	VahadaneStainNormalizer,         # stain normalizer
	NucleiExtractor,                 # nuclei detector 
	KNNGraphBuilder,                 # kNN graph builder
	ColorMergedSuperpixelExtractor,  # tissue detector
	DeepFeatureExtractor,            # feature extractor
	RAGGraphBuilder,                 # build graph
	AssignmnentMatrixBuilder         # assignment matrix 
)
from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization


class MGBuilding:
	def __init__(self, feat, cuda_id, mask_path):
		# define MG builders
		self._build_mg_builders(feat=feat, cuda_id=cuda_id)

		# define var to store image IDs that failed
		self.image_ids_failing = []

		self.melanocyte_path = mask_path

	def _build_mg_builders(self, feat, cuda_id):
		# 1 define feature extractor:
		# Options:
		#     DeepFeatureExtractor: Extract patches of 72x72 pixels around each nucleus centroid, then resize to 224 to match ResNet input size.
		#     HandCraftedFeat: Extract shape, size, depth features of each nucleus
		#     GanEncoder: Extract patches of 256x256 pixels around each nucleus centroid, then apply GAN encoder to extract features
		#     GanPlusHandCrafted: GAN features + Hand crafted features
		if feat=='DeepFeatureExtractor':
			self.melanocyte_feature_extractor = DeepFeatureExtractor(
				architecture='resnet34',
				patch_size=72,
				resize_size=224,
				cuda_id=cuda_id)
		elif feat=='HandCraftedFeat':
			self.melanocyte_feature_extractor = HandcraftedFeatureExtractor()
		elif feat=='GanEncoder':
			self.melanocyte_feature_extractor = GANFeatureExtractor(
				name='global_filtered_ngf32_256_local',
				which_epoch='180',
				patch_size=256,
				resize_size=512,
				batch_size=32,
				num_workers=16,
				cuda_id=cuda_id
				)
		elif feat=='GanPlusHandCrafted':
			#TODO: combine GAN and Handcrafted
			pass
			# self.melanocyte_feature_extractor = 
		else:
			raise

		# 2 define k-NN graph builder with k=15 and thresholding edges longer
		# than 50 pixels. Add image size-normalized centroids to the node features.
		# For e.g., resulting node features are 512 features from ResNet34 + 2
		# normalized centroid features.
		self.knn_graph_builder = KNNGraphBuilder(k=5, thresh=40, add_loc_feats=True)

	def _build_mg(self, image, image_id, image_size):
		melanocyte_map, melanocyte_centroids = self._melanocyte_extract(image_id, image_size)
		features = self.melanocyte_feature_extractor.process(image, melanocyte_map)
		graph = self.knn_graph_builder.process(melanocyte_map, features)
		return graph, melanocyte_map

	def _melanocyte_extract(self, image_id, image_size):
		case_id = image_id.split('_')[0]
		mask_path = glob.glob(os.path.join(self.melanocyte_path, '**', '{}*'.format(image_id)))
		assert len(mask_path) > 0

		# get melanocyte instance map
		nuclei_map = label(255*(np.array(Image.open(mask_path[0]).resize(image_size)) > 128).astype(np.uint8))

		# extract the centroid location in the instance map
		regions = regionprops(nuclei_map)
		instance_centroids = np.array([[int(round(x)) for x in region.centroid] for region in regions])

		return nuclei_map, instance_centroids

	def process(self, image_path, save_path, split):
		def parse_basename(image_path):
			image_label = image_path.split('/')[-2]
			bn = os.path.basename(image_path)
			# return bn.split('.')[0], int(image_label)
			case_id, slice_id = bn.split('_')[:2]
			if '.tif' in slice_id:
				slice_id = slice_id.replace('.tif', '')
			return '{}_{}'.format(case_id, slice_id), int(image_label)
		
		# 1 Load image & Get image id 
		image_id, image_label = parse_basename(image_path)
		image = Image.open(image_path)
		image_size = image.size
		image = np.array(image)
		mg_out = os.path.join(save_path, '{}.bin'.format(image_id))

		# 2 if file was not already created, then process
		if not os.path.isfile(mg_out):
			try:
				print('Processing Image: ', image_path)
				melanocyte_graph, melanocyte_map = self._build_mg(image, image_id, image_size)
				save_graphs(filename=mg_out, g_list=[melanocyte_graph], labels={"label": torch.tensor([image_label])})
			# visualizer = OverlayGraphVisualization(instance_visualizer=InstanceImageVisualization(instance_style="filled+outline"))
			# viz_cg = visualizer.process(canvas=image, graph=melanocyte_graph, instance_map=melanocyte_map)
			# viz_cg.save(os.path.join(save_path, '{}_graph.jpg'.format(image_id)))
			except:
				print('Warning: {} failed during cell graph generation.'.format(image_path))
				self.image_ids_failing.append(image_path)
				pass
		else:
			print('Image:', image_path, ' was already processed.')


if __name__ == '__main__':
	# 1. handle i/o
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_folder', type=str, help='path to the skin biopsy image.', default=None)
	parser.add_argument('--save_folder', type=str, help='path to save the cell graphs.', default=None)
	parser.add_argument('--mask_folder', type=str, help='path to melanocyte masks.', default=None)
	parser.add_argument('--start', type=int)
	parser.add_argument('--end', type=int)
	parser.add_argument('--cuda_id', type=str, default='0')
	args = parser.parse_args()

	os.makedirs(os.path.join(args.save_folder, 'cell_graphs'), exist_ok=True)

	# 2. generate melanocyte graphs one-by-one, will automatically run on GPU if available
	mg_builder = MGBuilding('GanEncoder', args.cuda_id, args.mask_folder)
	image_list = sorted(glob.glob(os.path.join(args.image_folder, '**', '**', '*.tif')))[args.start:args.end]
	print('Get {} images to process...'.format(len(image_list)))
	for image_path in image_list:
		split, class_name = image_path.split('/')[-3:-1]
		os.makedirs(os.path.join(args.save_folder, 'cell_graphs', split), exist_ok=True)
		os.makedirs(os.path.join(args.save_folder, 'cell_graphs', split, class_name), exist_ok=True)
		mg_builder.process(image_path, os.path.join(args.save_folder, 'cell_graphs', split, class_name), split)
	print('Failing IDs are:', mg_builder.image_ids_failing)
