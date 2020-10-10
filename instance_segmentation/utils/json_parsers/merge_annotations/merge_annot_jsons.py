import json
from shutil import copy
from config import *


def create_dir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)



def read_json(json_file_path):
	json_cont = None
	with open(json_file_path, 'r') as f:
		json_cont = json.load(f)
	return json_cont

def copy_file(src_path, dest_path):
	copy(src_path, dest_path)


def process_json(annot_json, input_dir):
	file_key_list = list(annot_json.keys())
	for file_key in file_key_list:
		image_file_name = annot_json[file_key]['filename']
		image_src_file_path = os.path.join(input_dir, image_file_name)

		if os.path.exists(image_src_file_path):
			image_dest_file_path = os.path.join(OUTPUT_DIR, image_file_name)
			copy_file(image_src_file_path, image_dest_file_path)
		else:
			# if file doesn't exceed, we can remove key
			print('Image file {} doesn\'t exists'.format(image_src_file_path))
			del annot_json[file_key]
	return annot_json



if __name__ == '__main__':

	if len(INPUT_DIR_LIST)>1:
		OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR)
		create_dir(OUTPUT_DIR)

		INPUT_DIR_BASE = os.path.join(ROOT_DIR, INPUT_DIR_LIST[0])

		print('For image dir: {}'.format(INPUT_DIR_LIST[0]))
		annot_json_base = read_json(os.path.join(INPUT_DIR_BASE, INPUT_DIRS_JSON))
		annot_json_base = process_json(annot_json_base, INPUT_DIR_BASE)

		for other_input_dir in INPUT_DIR_LIST[1:]: 
			print('For image dir: {}'.format(other_input_dir))
			# iterate starting from second
			input_dir = os.path.join(ROOT_DIR, other_input_dir)
			annot_json_rem = read_json(os.path.join(input_dir, INPUT_DIRS_JSON))
			annot_json_rem = process_json(annot_json_rem, input_dir)
			annot_json_base.update(annot_json_rem)

		output_json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON)
		with open(output_json_path, 'w') as outfile:
			json.dump(annot_json_base, outfile)
