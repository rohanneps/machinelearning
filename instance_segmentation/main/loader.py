import json
import os
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import utils
import tensorflow as tf

# silent tf verbose
tf.get_logger().setLevel('INFO')


class DamConfig(Config):
    """
    Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "farm_dam"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + FarmDam Class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 16

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1024

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    WEIGHT_DECAY = 0.001
    LEARNING_RATE = 0.0001
    VALIDATION_STEPS = 10



class FarmDamDataset(utils.Dataset):

    def load_dam(self, dataset_dir, subset):
        """
        Load a subset of the Farm Dam dataset.
        
        Params:
          dataset_dir: Root directory of the dataset.
          subset: Subset to load: train or val
        """
      
        # Add classes. We have only one class to add.
        self.add_class("farmdam", 1, "farmdam")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "images_region_annot.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)

            # Handling of inconsistent usage of annotaion format
            polygons = []

            for r in a['regions']:
                # idenifying type of region:
                if type(a['regions'])==dict:
                    for region_id in a['regions']:
                        polygons.append(a['regions'][region_id]['shape_attributes'])
                else:
                    # for list types
                    polygons.append(r['shape_attributes'])      
                                  

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "farmdam",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.

        Params:
          image_id: Id of the image
       
        Returns:
          masks: A bool array of shape [height, width, instance count] with
                 one mask per instance.
          class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a dam dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "farmdam":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """
        Return the path of the image as reference.
        """
        info = self.image_info[image_id]
        if info["source"] == "farmdam":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class InferenceConfig(DamConfig):
      # Set batch size to 1 since we'll be running inference on
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1