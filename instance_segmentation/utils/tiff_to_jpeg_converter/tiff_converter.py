from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

def create_dir(output_path):
	if not os.path.exists(output_path):
		os.mkdir(output_path)


def read_tiff_image(image_path):
	image = Image.open(image_path)
	out_image = image.convert("RGB")
	return out_image

def convert_tiff_files_in_folder(input_dir):
	for file in os.listdir(input_dir):
		tif_file_path = os.path.join(input_dir, file)
		out_image = read_tiff_image(tif_file_path)
		output_file_path = os.path.join(OUTPUT_DIR, '{}.jpeg'.format(file.split('.')[0]))
		out_image.save(output_file_path, "JPEG", quality=90)


if __name__ == '__main__':
	create_dir(OUTPUT_DIR)
	convert_tiff_files_in_folder(INPUT_DIR)
