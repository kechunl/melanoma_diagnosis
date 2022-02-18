import os
import shutil
import random
import glob

slice_class = {'1': ['903_1','903_2','908_1','908_2','908_3','908_4','909_4','909_5','909_6','909_8'],
				'2': ['902_3','902_4','910_8','910_9','910_10','910_11','911_2','911_3','911_4','911_5'],
				'3': ['901_1','901_2','901_4','901_6','912_2','912_5','913_1','913_2','913_3','913_4','913_5','913_6','914_0','914_2','914_5','914_6','915_16','915_19','915_18','915_21'],
				'4': ['900_2','914_1','914_4','915_17','915_20'],
				'5': ['908_0','909_9','916_5','916_6','916_7','916_8','917_0','917_1','917_5','10220_0']}

def data_split(root_dir, save_dir, train_scale=0.6, val_scale=0.1, test_scale=0.3):
	split_names = ['train', 'val', 'test']
	for split_name in split_names:
		split_path = os.path.join(save_dir, split_name)
		os.makedirs(split_path, exist_ok=True)
		for class_name in slice_class.keys():
			class_split_path = os.path.join(split_path, class_name)
			os.makedirs(class_split_path, exist_ok=True)

	for class_name, slice_list in slice_class.items():
		random.shuffle(slice_list)
		train_list = slice_list[:int(len(slice_list)*train_scale)]
		val_list = slice_list[int(len(slice_list)*train_scale):int(len(slice_list)*(train_scale+val_scale))]
		test_list = slice_list[int(len(slice_list)*(train_scale+val_scale)):]
		for subset, subset_list in zip(['train', 'val', 'test'], [train_list, val_list, test_list]):
			for slice_name in subset_list:
				slice_path = glob.glob(os.path.join(root_dir, '**', '{}_x10_z0.tif'.format(slice_name)))[0]
				shutil.copy(slice_path, os.path.join(save_dir, subset, class_name, os.path.basename(slice_path)))


if __name__ == '__main__':
	root_dir = '/projects/patho4/Kechun/diagnosis/dataset/HE/x10'
	save_dir = '/projects/patho4/Kechun/diagnosis/dataset/slice'
	os.makedirs(save_dir, exist_ok=True)
	data_split(root_dir, save_dir)
