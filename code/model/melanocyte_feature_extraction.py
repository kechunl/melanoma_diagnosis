import copy
import math
import os
import glob
import warnings
from abc import abstractmethod
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Tuple, Union
import yaml
import sys
from argparse import Namespace
import pdb

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
sys.path.append('/projects/patho4/Kechun/diagnosis/histocartography/')
from histocartography.preprocessing.feature_extraction import FeatureExtractor, PatchFeatureExtractor, InstanceMapPatchDataset
from histocartography.preprocessing.tissue_mask import GaussianTissueMask
from histocartography.utils import dynamic_import_from
from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.measure import regionprops
from skimage.morphology import disk
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

sys.path.append('/projects/patho4/Kechun/melanocyte_detection/GAN/End-to-End pix2pixHD/')
from models.models import create_model

class GANFeatureExtractor(FeatureExtractor):
	"""Helper class to extract features using GAN generator from instance maps"""
	def __init__(
		self, 
		name: str, 
		which_epoch: str,
		checkpoint_dir: str = '/projects/patho4/Kechun/melanocyte_detection/GAN/End-to-End pix2pixHD/checkpoints',
		downsample_factor: int = 1,
		patch_size: int = 256,
		resize_size: int = 512,
		batch_size: int = 32,
		fill_value: int = 255,
		num_workers: int = 0,
		cuda_id: str = '0',
		) -> None:
		"""
		Create a GAN feature extractor. The feature is generated from GAN + RCNN which are trained in melanocyte detection.

		Args:
			name (str): Name of the GAN model to use.
			which_epoch (str): prefix of the checkpoint.
		"""
		self.checkpoint_dir = checkpoint_dir
		self.model_dir = os.path.join(self.checkpoint_dir, name)
		self.which_epoch = which_epoch
		self.patch_size = patch_size
		self.resize_size = resize_size
		self.downsample_factor = downsample_factor

		# Handle GPU
		cuda = torch.cuda.is_available()
		self.cuda_id = cuda_id
		self.device = torch.device('cuda:'+cuda_id if cuda else 'cpu')
		self._preprocess_architecture()
		self.fill_value = fill_value
		self.batch_size = batch_size
		self.num_workers = num_workers
		if self.num_workers in [0, 1]:
			torch.set_num_threads(1)

	def _preprocess_architecture(self):
		"""Preprocess architecture to load the model"""
		with open(os.path.join(self.model_dir, 'opt.yaml'), "r") as f:
			params = yaml.load(f, Loader=yaml.FullLoader)
		params['which_epoch'] = self.which_epoch
		params['isTrain'] = False
		params['load_pretrain'] = None
		params['checkpoints_dir'] = self.checkpoint_dir
		params['gpu_device'] = 'cuda:' + self.cuda_id
		self.model = create_model(Namespace(**params))
		self.model.eval()

	def _collate_patches(self, batch):
		"""Patch collate function"""
		instance_indices = [item[0] for item in batch]
		patches = [item[1] for item in batch]
		patches = torch.stack(patches)
		proposals = [item[2] for item in batch]
		return instance_indices, patches, proposals

	def _extract_features(self,
		input_image: np.ndarray,
		instance_map: np.ndarray,
		)-> torch.Tensor:
		"""
		Extract features for a given RGB image and its extracted instance map.

		Args:
			input_image (np.ndarray): RGB input image.
			instance_map (np.ndarray): Extracted instance map.
		Returns:
			torch.Tensor: Extracted features of shape [#instances, #features]
		"""
		if self.downsample_factor != 1:
			input_image = self._downsample(input_image, self.downsample_factor)
			instance_map = self._downsample(instance_map, self.downsample_factor)
		
		image_dataset = CenterPatchDataset(
			image=input_image,
			instance_map=instance_map,
			patch_size=self.patch_size,
			resize_size=self.resize_size,
			fill_value=self.fill_value
			)
		image_loader = DataLoader(
			image_dataset,
			shuffle=False,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			collate_fn=self._collate_patches
			)
		features = torch.empty(size=(len(image_dataset.properties), 256), dtype=torch.float32, device=self.device)
		for instance_indices, patches, proposals in tqdm(image_loader, total=len(image_loader)):
			emb = self._get_features(patches, proposals) # Batch x 256
			for j, key in enumerate(instance_indices):
				features[key, :] = emb[j]
		return features.cpu().detach()

	def _get_features(self, patch: torch.Tensor, proposals: List[torch.Tensor]) -> torch.Tensor:
		"""
		Computes the embedding of a normalized image input.

		Args:
			image (torch.Tensor): Normalized image input. Batch x channels x H x W

		Returns:
			torch.Tensor: Embedding of image.
		"""
		patch = patch.to(self.device)
		proposals = [p.to(self.device) for p in proposals]
		image_sizes = [(self.resize_size, self.resize_size)] * patch.shape[0]
		with torch.no_grad():
			_, img_features = self.model.netG.forward(patch)
			img_features = self.model.fpn(img_features) # B x 32 x (16x16, 32x32, 64x64, 128x128, 256x256)
			instance_features = self.model.roi_heads.mask_roi_pool(img_features, proposals, image_sizes)
			instance_features = self.model.roi_heads.mask_head(instance_features) # B x 256 x 14 x 14
		return torch.mean(instance_features.cpu().detach(), dim=(2,3))


class CenterPatchDataset(Dataset):
	"""Helper class to use a give image and extracted instance maps as a dataset"""
	def __init__(
		self,
		image: np.ndarray,
		instance_map: np.ndarray,
		patch_size: int,
		resize_size: int = None,
		fill_value: Optional[int] = 255,
		) -> None:
		"""
		Create a dataset for a given image and extracted instance map with desired patches
		of (patch_size, patch_size, 3). 

		Args:
			image (np.ndarray): RGB input image.
			instance map (np.ndarray): Extracted instance map.
			patch_size (int): Desired size of patch.
			stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
			resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
							   patches of size patch_size are provided to the network. Defaults to None.
			fill_value (Optional[int]): Value to fill outside the instance maps. Defaults to 255.
		"""
		self.image = image
		self.instance_map = instance_map
		self.patch_size = patch_size
		self.fill_value = fill_value
		self.resize_size = resize_size

		self.image = np.pad(
			self.image,
			(
				(self.patch_size, self.patch_size),
				(self.patch_size, self.patch_size),
				(0, 0),
			),
			mode="constant",
			constant_values=fill_value,
		)
		self.instance_map = np.pad(
			self.instance_map,
			((self.patch_size, self.patch_size), (self.patch_size, self.patch_size)),
			mode="constant",
			constant_values=0,
		)

		self.patch_size_2 = int(self.patch_size // 2)
		self.properties = regionprops(self.instance_map)
		self.patch_coordinates = []
		self.patch_region_count = []
		self.patch_instance_ids = []
		self.patch_instance_bbox = []

		basic_transforms = [transforms.ToPILImage()]
		if self.resize_size is not None:
			basic_transforms.append(transforms.Resize(self.resize_size))
		basic_transforms.append(transforms.ToTensor())
		self.dataset_transform = transforms.Compose(basic_transforms)

		self._precompute()

	def _precompute(self):
		"""Precompute instance-wise patch information for all instances in the input image"""
		for region_count, region in enumerate(self.properties):
			
			# Extract centroid
			center_y, center_x = region.centroid
			center_x = int(round(center_x))
			center_y = int(round(center_y))

			# Extract bounding box
			min_y, min_x, max_y, max_x = region.bbox
			min_y = max(min_y, center_y-self.patch_size_2)
			min_x = max(min_x, center_x-self.patch_size_2)
			max_y = min(max_y, center_y+self.patch_size_2)
			max_x = min(max_x, center_x+self.patch_size_2)

			# Extract patch information around the centroid
			self.patch_coordinates.append([center_x - self.patch_size_2, center_y - self.patch_size_2])
			self.patch_region_count.append(region_count)
			self.patch_instance_ids.append(region.label)
			self.patch_instance_bbox.append([min_y, min_x, max_y, max_x])
		# self.patch_coordinates = np.array(self.patch_coordinates)
		# self.patch_region_count = np.array(self.patch_region_count)
		# self.patch_instance_ids = np.array(self.patch_instance_ids)
		# self.patch_instance_bbox = np.array(self.patch_instance_bbox)

	def _get_patch(self, loc: list, region_id: int = None) -> np.ndarray:
		"""
		Extract patch from image.

		Args:
			loc (list): Top-left (x,y) coordinate of a patch.
			region_id (int): Index of the region being processed. Defaults to None.
		"""
		min_x = loc[0]
		min_y = loc[1]
		max_x = min_x + self.patch_size
		max_y = min_y + self.patch_size

		patch = copy.deepcopy(self.image[min_y:max_y, min_x:max_x])

		return patch

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
		"""
		Loads an image for a give patch index.

		Args:
			index (int): Patch index.

		Returns:
			Tuple[int, torch.Tensor]: instance_index, image as tensor.
		"""
		patch = self._get_patch(self.patch_coordinates[index], self.patch_instance_ids[index])
		patch = self.dataset_transform(patch)
		proposal = torch.FloatTensor(self.resize_size / self.patch_size * np.array(self.patch_instance_bbox[index]))[None, :]
		return self.patch_region_count[index], patch, proposal

	def __len__(self) -> int:
		"""
		Returns the length of the dataset.

		Returns:
			int: Length of the dataset
		"""
		return len(self.patch_coordinates)


class DeepFeatureExtractor(FeatureExtractor):
	"""Helper class to extract deep features from instance maps"""

	def __init__(
		self,
		architecture: str,
		patch_size: int,
		resize_size: int = None,
		stride: int = None,
		downsample_factor: int = 1,
		normalizer: Optional[dict] = None,
		batch_size: int = 32,
		fill_value: int = 255,
		num_workers: int = 0,
		verbose: bool = False,
		with_instance_masking: bool = False,
		cuda_id: str = '0',
		**kwargs,
	) -> None:
		"""
		Create a deep feature extractor.

		Args:
			architecture (str): Name of the architecture to use. According to torchvision.models syntax.
			patch_size (int): Desired size of patch.
			resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
							   patches of size patch_size are provided to the network. Defaults to None.
			stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
			downsample_factor (int): Downsampling factor for image analysis. Defaults to 1.
			normalizer (dict): Dictionary of channel-wise mean and standard deviation for image
							   normalization. If None, using ImageNet normalization factors. Defaults to None.
			batch_size (int): Batch size during processing of patches. Defaults to 32.
			fill_value (int): Constant pixel value for image padding. Defaults to 255.
			num_workers (int): Number of workers in data loader. Defaults to 0.
			verbose (bool): tqdm processing bar. Defaults to False.
			with_instance_masking (bool): If pixels outside instance should be masked. Defaults to False.
		"""
		self.architecture = self._preprocess_architecture(architecture)
		self.patch_size = patch_size
		self.resize_size = resize_size
		if stride is None:
			self.stride = patch_size
		else:
			self.stride = stride
		self.downsample_factor = downsample_factor
		self.with_instance_masking = with_instance_masking
		self.verbose = verbose
		if normalizer is not None:
			self.normalizer = normalizer.get("type", "unknown")
		else:
			self.normalizer = None
		super().__init__(**kwargs)

		# Handle GPU
		cuda = torch.cuda.is_available()
		self.device = torch.device("cuda:"+cuda_id if cuda else "cpu")

		if normalizer is not None:
			self.normalizer_mean = normalizer.get("mean", [0, 0, 0])
			self.normalizer_std = normalizer.get("std", [1, 1, 1])
		else:
			self.normalizer_mean = [0.485, 0.456, 0.406]
			self.normalizer_std = [0.229, 0.224, 0.225]
		self.patch_feature_extractor = PatchFeatureExtractor(
			architecture, device=self.device
		)
		self.fill_value = fill_value
		self.batch_size = batch_size
		self.architecture_unprocessed = architecture
		self.num_workers = num_workers
		if self.num_workers in [0, 1]:
			torch.set_num_threads(1)

	def _collate_patches(self, batch):
		"""Patch collate function"""
		instance_indices = [item[0] for item in batch]
		patches = [item[1] for item in batch]
		patches = torch.stack(patches)
		return instance_indices, patches

	def _extract_features(
		self,
		input_image: np.ndarray,
		instance_map: np.ndarray,
		transform: Optional[Callable] = None
	) -> torch.Tensor:
		"""
		Extract features for a given RGB image and its extracted instance_map.

		Args:
			input_image (np.ndarray): RGB input image.
			instance_map (np.ndarray): Extracted instance_map.
			transform (Callable): Transform to apply. Defaults to None.
		Returns:
			torch.Tensor: Extracted features of shape [nr_instances, nr_features]
		"""
		if self.downsample_factor != 1:
			input_image = self._downsample(input_image, self.downsample_factor)
			instance_map = self._downsample(
				instance_map, self.downsample_factor)

		image_dataset = InstanceMapPatchDataset(
			image=input_image,
			instance_map=instance_map,
			resize_size=self.resize_size,
			patch_size=self.patch_size,
			stride=self.stride,
			fill_value=self.fill_value,
			mean=self.normalizer_mean,
			std=self.normalizer_std,
			transform=transform,
			with_instance_masking=self.with_instance_masking,
		)
		image_loader = DataLoader(
			image_dataset,
			shuffle=False,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			collate_fn=self._collate_patches
		)
		features = torch.empty(
			size=(
				len(image_dataset.properties),
				self.patch_feature_extractor.num_features,
			),
			dtype=torch.float32,
			device=self.device,
		)
		embeddings = dict()
		for instance_indices, patches in tqdm(
			image_loader, total=len(image_loader), disable=not self.verbose
		):
			emb = self.patch_feature_extractor(patches)
			for j, key in enumerate(instance_indices):
				if key in embeddings:
					embeddings[key][0] += emb[j]
					embeddings[key][1] += 1
				else:
					embeddings[key] = [emb[j], 1]

		for k, v in embeddings.items():
			features[k, :] = v[0] / v[1]

		return features.cpu().detach()


class HandcraftedFeatureExtractor(FeatureExtractor):
	"""Helper class to extract handcrafted features from instance maps"""
	def _extract_features(self, input_image: np.ndarray, instance_map: np.ndarray) -> torch.Tensor:
		"""
		Extract handcrafted features from the input_image in the defined instance_map regions.

		Args:
			input_image (np.array): Original RGB Image.
			instance_map (np.array): Extracted instance_map. Different regions have different int values,
									 the background is defined to have value 0 and is ignored.

		Returns:
			torch.Tensor: Extracted shape, color and texture features:
						  Shape:   area, convex_area, eccentricity, equivalent_diameter, euler_number, extent, filled_area,
								   major_axis_length, minor_axis_length, orientation, perimiter, solidity;
						  Texture: glcm_contrast, glcm_dissililarity, glcm_homogeneity, glcm_energy, glcm_ASM, glcm_dispersion
								   (glcm = grey-level co-occurance matrix);
						  Crowdedness: mean_crowdedness, std_crowdedness

		"""
		node_feat = []
		img_gray = np.array(Image.fromarray(input_image).convert('L'))

		regions = regionprops(instance_map)

		# pre-extract centroids to compute crowdedness
		centroids = [r.centroid for r in regions]
		all_mean_crowdedness, all_std_crowdedness = self._compute_crowdedness(
			centroids)

		for region_id, region in enumerate(regions):

			sp_mask = instance_map[region['bbox'][0]:region['bbox'][2], region['bbox'][1]:region['bbox'][3]] == region['label'] 
			sp_gray = img_gray[region['bbox'][0]:region['bbox'][2], region['bbox'][1]:region['bbox'][3]] * sp_mask

			# Compute using mask [16 features]
			area = region["area"]
			convex_area = region["convex_area"]
			eccentricity = region["eccentricity"]
			equivalent_diameter = region["equivalent_diameter"]
			# euler_number = region["euler_number"]
			extent = region["extent"]
			filled_area = region["filled_area"]
			major_axis_length = region["major_axis_length"]
			minor_axis_length = region["minor_axis_length"]
			orientation = region["orientation"]
			perimeter = region["perimeter"]
			solidity = region["solidity"]
			convex_hull_perimeter = self._compute_convex_hull_perimeter(sp_mask)
			roughness = convex_hull_perimeter / perimeter
			shape_factor = 4 * np.pi * area / convex_hull_perimeter ** 2
			ellipticity = minor_axis_length / major_axis_length
			roundness = (4 * np.pi * area) / (perimeter ** 2)

			feats_shape = [
				area,
				convex_area,
				eccentricity,
				equivalent_diameter,
				# euler_number,
				extent,
				filled_area,
				major_axis_length,
				minor_axis_length,
				orientation,
				perimeter,
				solidity,
				roughness,
				shape_factor,
				ellipticity,
				roundness,
			]

			# GLCM texture features (gray color space) [6 features]
			glcm = greycomatrix(sp_gray, [1], [0])
			# Filter out the first row and column
			filt_glcm = glcm[1:, 1:, :, :]

			glcm_contrast = greycoprops(filt_glcm, prop="contrast")
			glcm_contrast = glcm_contrast[0, 0]
			glcm_dissimilarity = greycoprops(filt_glcm, prop="dissimilarity")
			glcm_dissimilarity = glcm_dissimilarity[0, 0]
			glcm_homogeneity = greycoprops(filt_glcm, prop="homogeneity")
			glcm_homogeneity = glcm_homogeneity[0, 0]
			glcm_energy = greycoprops(filt_glcm, prop="energy")
			glcm_energy = glcm_energy[0, 0]
			glcm_ASM = greycoprops(filt_glcm, prop="ASM")
			glcm_ASM = glcm_ASM[0, 0]
			glcm_dispersion = np.std(filt_glcm)

			feats_texture =  [
				glcm_contrast,
				glcm_dissimilarity,
				glcm_homogeneity,
				glcm_energy,
				glcm_ASM,
				glcm_dispersion,
			]

			feats_crowdedness = [
				all_mean_crowdedness[region_id],
				all_std_crowdedness[region_id],
			]

			#TODO Depth feature (from epidermis or dermis)

			sp_feats = feats_shape + feats_texture + feats_crowdedness # 15+6+2=23 features
			features = np.hstack(sp_feats)
			node_feat.append(features)

		node_feat = np.vstack(node_feat)
		return torch.Tensor(node_feat)


	@staticmethod
	def _compute_crowdedness(centroids, k=10):
		n_centroids = len(centroids)
		if n_centroids < 3:
			mean_crow = np.array([[0]] * n_centroids)
			std_crow = np.array([[0]] * n_centroids)
			return mean_crow, std_crow
		if n_centroids < k:
			k = n_centroids - 2
		dist = euclidean_distances(centroids, centroids)
		idx = np.argpartition(dist, kth=k + 1, axis=-1)
		x = np.take_along_axis(dist, idx, axis=-1)[:, : k + 1] # len(centroids) x (k+1)
		std_crowd = np.reshape(np.std(x, axis=1), newshape=(-1, 1))
		mean_crow = np.reshape(np.mean(x, axis=1), newshape=(-1, 1))
		return mean_crow, std_crowd

	def _compute_convex_hull_perimeter(self, sp_mask):
		"""Compute the perimeter of the convex hull induced by the input mask."""
		if cv2.__version__[0] == "3":
			_, contours, _ = cv2.findContours(
				np.uint8(sp_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
			)
		elif cv2.__version__[0] == "4":
			contours, _ = cv2.findContours(
				np.uint8(sp_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
			)
		hull = cv2.convexHull(contours[0])
		convex_hull_perimeter = cv2.arcLength(hull, True)

		return convex_hull_perimeter