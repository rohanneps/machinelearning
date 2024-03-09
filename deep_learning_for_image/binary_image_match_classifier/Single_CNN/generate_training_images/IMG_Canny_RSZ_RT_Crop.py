import os
from PIL import Image
from PIL import ImageFilter

class Process_images(object):


    def load_images_from_directory(self, img_directory):
        image_file_list = []
        for root, dir, files in os.walk(img_directory):
            for file_name in files:
                image_file = file_name
                image_file_list.append(image_file)
        return image_file_list

    def imagecreatefromjpeg(self, img_directory, img_file):
        img = Image.open(os.path.join(img_directory,img_file))
        return img

    def resizeimage(self,width,height,image):
        return image.resize((width, height), Image.ANTIALIAS)

    def imagerotate(self, image, degrees):
        Rotated = image.rotate(degrees)
        return Rotated

    def saveimagejpeg(self, image_save_dir, image, image_file_name):
        saved = image.save(os.path.join(image_save_dir ,image_file_name))
        return saved

    def rotate_image_and_save_to_features(self, img_directory, img_file, degree, img_save_dir):
        source = self.imagecreatefromjpeg(img_directory, img_file)
        rotate = self.imagerotate(source, degree)
        saved = self.saveimagejpeg(img_save_dir, rotate, str(degree)+'_'+img_file)
        return rotate

    def get_rotated_images(self, img_directory, img_save_dir, degrees,file_name):
        rotated = self.rotate_image_and_save_to_features(img_directory, file_name, degrees, img_save_dir)
        return rotated
    def get_monochrome_only(self,image):
        return image.convert("L")

    def monochrome(self, img_directory, img_save_dir, image_file_name):
        im = self.imagecreatefromjpeg(img_directory, image_file_name)
        gray_image = im.convert("L")
        saved = self.saveimagejpeg(img_save_dir, gray_image, image_file_name)
        return gray_image

    def get_monochromed_images(self, img_directory, img_save_dir,file_name):
        monochromed = self.monochrome(img_directory, img_save_dir, file_name)
        return monochromed

    def getimagesize(self, image):
        return image.size

    def merge_images(self, image1, image2, img_save_dir, img_file):
        width1, height1 = self.getimagesize(image1)
        width2, height2 = self.getimagesize(image2)
        result_width = width1 + width2
        result_height = max(height1, height2)

        result = Image.new("RGB", (result_width, result_height),(255,255,255))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(width1, 0))
        saved = self.saveimagejpeg(img_save_dir, result, img_file)
        return saved

    def resize_images(self, img_directory, img_save_dir, img_file_name, width, height):
        src = self.imagecreatefromjpeg(img_directory, img_file_name)
        resized = src.resize((width, height), Image.ANTIALIAS)
        saved = self.saveimagejpeg(img_save_dir, resized, img_file_name)
        return resized

    def get_merged_images(self, image1, image2, img_save_dir, img_file):
        merged = self.merge_images(image1, image2, img_save_dir, img_file)
        return merged

    def get_resized_images(self, img_directory, img_save_dir, img_file_name, width, height):
        resized = self.resize_images(img_directory, img_save_dir, img_file_name, width, height)
        return resized

    def detect_canny_edges(self, img_directory, img_save_dir, img_file_name):
        src = self.imagecreatefromjpeg(img_directory, img_file_name)
        Edge_extracted = src.filter(ImageFilter.FIND_EDGES)
        saved = self.saveimagejpeg(img_save_dir, Edge_extracted, img_file_name)
        return Edge_extracted

    def get_canny_edges(self, img_directory, img_save_dir, img_file_name):
        cannied = self.detect_canny_edges(img_directory, img_save_dir, img_file_name)
        return cannied

    def get_cannyonly(self,image):
        return image.filter(ImageFilter.FIND_EDGES)

    def get_croponly(self,image,x1,y1,x2,y2):
        return image.crop((x1, y1, x2, y2))


    def crop_images(self,img_directory,  img_file_name,x1,y1,x2,y2,image_save_dir):
        #cannied=self.get_canny_edges(img_directory, img_save_dir, img_file_name)
        src = self.imagecreatefromjpeg(img_directory, img_file_name)
        cropped=src.crop((x1,y1,x2,y2))
        saved = self.saveimagejpeg(image_save_dir, cropped, img_file_name)
        return cropped

    def get_cropped_image(self, img_directory, img_file_name,x1,y1,x2,y2,image_save_dir):
        cropped = self.crop_images(img_directory, img_file_name,x1,y1,x2,y2,image_save_dir)
        return cropped

    def match_images(self,pri_load_images_from_directory,sec_load_images_from_directory):
        pri=self.load_images_from_directory(pri_load_images_from_directory)
        sec = self.load_images_from_directory(sec_load_images_from_directory)
        for file1 in pri:
            for file2 in sec:
                if file1==file2:
                    return "Matched"
                else: return "UnMatched"

    def get_matched_images_dir(self,pri_load_images_from_directory,sec_load_images_from_directory):
        matched_unmatched=self.match_images(pri_load_images_from_directory,sec_load_images_from_directory)
        return matched_unmatched

    def get_match_images_file(self,file1,file2):
        if file1==file2:
            return "Matched"
        else: return "UnMatched"

    def split_img_from_dir(self,img_dir):
        img_files=self.load_images_from_directory(img_dir)
        splitted=[]
        for imgf in img_files:
            img=self.imagecreatefromjpeg(imgf.split('/')[0],imgf.split('/')[1])
            splitted.append(img.split())
        return splitted

    def split_img(self,image):
        return image.split()

    def get_splitted_img(self, img_directory, img_save_dir, img_file_name):
        splitted = self.split_img_from_dir(img_directory)
        for split1 in splitted:
            saved=self.saveimagejpeg(img_save_dir,split1,img_file_name)
        return saved




