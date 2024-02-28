import os
import Augmentor
import argparse


OUTPUT_DIR = "./augmented_images"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment Images for deep learning Model training. Images are generated in the same folder as input."
    )
    parser.add_argument("--imgs_dir_path", help="Image Directory path here", type=str)
    parser.add_argument(
        "--num_images",
        help="Number of Images to be augmented for each category",
        type=int,
    )
    args = parser.parse_args()

    image_dir = args.imgs_dir_path
    num_aug_images = args.num_images

    for roots, dirs, files in os.walk(image_dir):

        for dir_ in dirs:

            dir_full_path = os.path.join(image_dir, dir_)

            for subroots, subdirs, subfiles in os.walk(dir_full_path):

                print(dir_)
                print(len(subfiles))

                p = Augmentor.Pipeline(dir_full_path, output_directory=OUTPUT_DIR)
                p.rotate90(probability=0.3)
                p.rotate270(probability=0.3)
                p.flip_left_right(probability=0.3)
                p.flip_top_bottom(probability=0.3)
                p.skew_tilt(probability=0.3)
                p.skew_top_bottom(probability=0.3)
                p.random_distortion(
                    probability=0.5, grid_width=4, grid_height=4, magnitude=8
                )
                p.crop_random(probability=1, percentage_area=0.5)
                p.sample(num_aug_images)
