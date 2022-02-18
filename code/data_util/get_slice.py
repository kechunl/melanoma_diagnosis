import os, shutil, argparse


def read_txt(txt_path, delimiter=';'):
	txt_file = open(txt_path, 'r')
	Lines = txt_file.readlines()
 	
	results = []
	for line in Lines:
		results.append(line.strip().split(delimiter))
	return results

def main(args):
	os.makedirs(args.save_folder, exist_ok=True)
	os.makedirs(os.path.join(args.save_folder, args.split), exist_ok=True)
	slice_info_list = read_txt(args.txt_file)
	failing = []

	for slice_info in slice_info_list:
		try:
			path = slice_info[0]
			diag_class = path.split('/')[-2]
			os.makedirs(os.path.join(args.save_folder, args.split, diag_class), exist_ok=True)
			if slice_info[2] == '0':
				shutil.copy(path, os.path.join(args.save_folder, args.split, diag_class, os.path.basename(path)))
		except:
			failing.append(slice_info[0])
	print('failing: ', failing)



if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--save_folder', type=str, help='path to save the skin biopsy slice image.', default=None)
	argparser.add_argument('--txt_file', type=str, help='slice info text file path', default=None)
	args = argparser.parse_args()
	args.split = os.path.basename(args.txt_file).split('.')[0]
	main(args)
