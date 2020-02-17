import os,errno
from IMG_Canny_RSZ_RT_Crop import Process_images as labelimages
import sys
from imutils import paths
import argparse

def image_operations(input_image_folder, output_image_folder):

	imgfunc=labelimages()
	imagePaths = sorted(list(paths.list_images(input_image_folder)))

	for imagePath in imagePaths:
		# print(imagePath)
		label = imagePath.split('/')[-2]
		filename=imagePath.split('/')[-1]

		print(input_image_folder)

		for root, dirnames, filenames in os.walk(input_image_folder):
			unmatched_counter = 0

			for fn in filenames:
				print("Matched>",fn,root,filename,input_image_folder)
			
				if fn==filename and input_image_folder==root:
					print(root,input_image_folder,fn,filename)
					
					matchedfolder= os.path.join(output_image_folder,'Matched')
					try:
						os.makedirs(matchedfolder)
					except OSError as e:
						if e.errno != errno.EEXIST:
							raise
					
					img1=imgfunc.imagecreatefromjpeg(root,filename)
					img2 = imgfunc.imagecreatefromjpeg(root,filename)
					split1=imgfunc.split_img(img1)
					try:
						r1, g1, b1 = split1
					except:
						r1, g1, b1, a1 = split1
					split2 = imgfunc.split_img(img2)

					try:
						r2, g2, b2 = split2
					except:
						r2, g2, b2,a2 = split2

					merged = imgfunc.get_merged_images(r1, r2,matchedfolder,'Splitted_1' + label + '_' + filename)
					merged = imgfunc.get_merged_images(g1, g2,matchedfolder,'Splitted_2' + label + '_' + filename)

					merged = imgfunc.get_merged_images(b1, b2,matchedfolder,'Splitted_3' + label + '_' + filename)

					rotated1=imgfunc.imagerotate(img1,90)
					rotated2 = imgfunc.imagerotate(img1, 180)
					rotated3 = imgfunc.imagerotate(img1, 270)
					rotated4 = imgfunc.imagerotate(img2, 90)
					rotated5 = imgfunc.imagerotate(img2, 180)
					rotated6 = imgfunc.imagerotate(img2, 270)
					rotated10 = imgfunc.imagerotate(img2, 0)
					rotated0 = imgfunc.imagerotate(img1, 0)
					merged = imgfunc.get_merged_images(rotated0, rotated10,matchedfolder,'rt_0_' + label + '_' + filename)
					merged = imgfunc.get_merged_images(rotated1, rotated4, matchedfolder,'rt_90_'+label + '_' + filename)
					merged = imgfunc.get_merged_images(rotated2, rotated5, matchedfolder,'rt_180_' + label + '_' + filename)
					merged = imgfunc.get_merged_images(rotated3, rotated6, matchedfolder,'rt_270_' + label + '_' + filename)



					resized1=imgfunc.resizeimage(200,200,img1)
					resized2 = imgfunc.resizeimage(100, 100, img1)
					resized3 = imgfunc.resizeimage(150, 150, img1)
					resized4 = imgfunc.resizeimage(50, 50, img1)
					resized5 = imgfunc.resizeimage(200, 200, img2)
					resized6 = imgfunc.resizeimage(100, 100, img2)
					resized7 = imgfunc.resizeimage(150, 150, img2)
					resized8 = imgfunc.resizeimage(50, 50, img2)
					merged = imgfunc.get_merged_images(resized1, resized5, matchedfolder,'rsz_200_'+label + '_' + filename)
					merged = imgfunc.get_merged_images(resized2, resized6, matchedfolder,'rsz_100_' + label + '_' + filename)
					merged = imgfunc.get_merged_images(resized3, resized7, matchedfolder,'rsz_150_' + label + '_' + filename)
					merged = imgfunc.get_merged_images(resized4, resized8, matchedfolder,'rsz_50_' + label + '_' + filename)

					cannied1=imgfunc.get_cannyonly(img1)
					cropped1=imgfunc.get_croponly(img1,0,0,100,100)
					monochrom1=imgfunc.get_monochrome_only(img1)

					cannied2 = imgfunc.get_cannyonly(img2)
					cropped2 = imgfunc.get_croponly(img2, 0, 0, 100, 100)
					monochrom2 = imgfunc.get_monochrome_only(img2)
					merged = imgfunc.get_merged_images(cannied1, cannied2, matchedfolder,'can_'+label + '_' + filename)
					merged = imgfunc.get_merged_images(cropped1, cropped2, matchedfolder,  'crop_'+label + '_' + filename)
					merged = imgfunc.get_merged_images(monochrom1, monochrom2, matchedfolder,'mono_'+label + '_' + filename)
					
				elif fn!=filename and input_image_folder==root:
					print("UnMatched>",root,input_image_folder,fn,filename)
					
					unmatchedfolder= os.path.join(output_image_folder,'UnMatched')
					try:
						os.makedirs(unmatchedfolder)
					except OSError as e:
						if e.errno != errno.EEXIST:
							raise

					if unmatched_counter < 20:
					
						img1=imgfunc.imagecreatefromjpeg(root,filename)
						img2 = imgfunc.imagecreatefromjpeg(root,fn)
						
						
						split1 = imgfunc.split_img(img1)
						try:
							r1, g1, b1 = split1
						except:
							r1, g1, b1, a1 = split1
						split2 = imgfunc.split_img(img2)
						try:
							r2, g2, b2 = split2
						except:
							r2, g2, b2,a2 = split2
						merged = imgfunc.get_merged_images(r1, r2,unmatchedfolder,'Splitted_1' + label + '_' + fn.split('.')[0]+'_'+filename)
						
						merged = imgfunc.get_merged_images(g1, g2,unmatchedfolder,'Splitted_2' + label + '_' + fn.split('.')[0]+'_'+filename)

						merged = imgfunc.get_merged_images(b1, b2,unmatchedfolder,'Splitted_2' + label + '_' + fn.split('.')[0]+'_'+filename)
						rotated1 = imgfunc.imagerotate(img1, 90)
						rotated2 = imgfunc.imagerotate(img1, 180)
						rotated3 = imgfunc.imagerotate(img1, 270)
						rotated4 = imgfunc.imagerotate(img2, 90)
						rotated5 = imgfunc.imagerotate(img2, 180)
						rotated6 = imgfunc.imagerotate(img2, 270)
						rotated10 = imgfunc.imagerotate(img2, 0)
						rotated0 = imgfunc.imagerotate(img1, 0)
						merged = imgfunc.get_merged_images(rotated0, rotated10,unmatchedfolder,'rt_0_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(rotated1, rotated4, unmatchedfolder,'rt_90_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(rotated2, rotated5, unmatchedfolder,'rt_180_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(rotated3, rotated6, unmatchedfolder,'rt_270_' + label + '_' + fn.split('.')[0]+'_'+filename)

						resized1 = imgfunc.resizeimage(200, 200, img1)
						resized2 = imgfunc.resizeimage(100, 100, img1)
						resized3 = imgfunc.resizeimage(150, 150, img1)
						resized4 = imgfunc.resizeimage(50, 50, img1)
						resized5 = imgfunc.resizeimage(200, 200, img2)
						resized6 = imgfunc.resizeimage(100, 100, img2)
						resized7 = imgfunc.resizeimage(150, 150, img2)
						resized8 = imgfunc.resizeimage(50, 50, img2)
						merged = imgfunc.get_merged_images(resized1, resized5, unmatchedfolder,'rsz_200_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(resized2, resized6, unmatchedfolder,'rsz_100_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(resized3, resized7, unmatchedfolder,'rsz_150_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(resized4, resized8, unmatchedfolder,'rsz_50_' + label + '_' + fn.split('.')[0]+'_'+filename)

						cannied1 = imgfunc.get_cannyonly(img1)
						cropped1 = imgfunc.get_croponly(img1, 0, 0, 100, 100)
						monochrom1 = imgfunc.get_monochrome_only(img1)

						cannied2 = imgfunc.get_cannyonly(img2)
						cropped2 = imgfunc.get_croponly(img2, 0, 0, 100, 100)
						monochrom2 = imgfunc.get_monochrome_only(img2)
						merged = imgfunc.get_merged_images(cannied1, cannied2, unmatchedfolder,'can_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(cropped1, cropped2, unmatchedfolder,'crop_' + label + '_' + fn.split('.')[0]+'_'+filename)
						merged = imgfunc.get_merged_images(monochrom1, monochrom2, unmatchedfolder,'mono_' + label + '_' + fn.split('.')[0]+'_'+filename)
						unmatched_counter += 1
					else:
						continue
					
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_img_folder", help="Input images folder", type=str)
	parser.add_argument("--output_img_folder", help="Output directory where images are to be stored categorically", type=str) 
	args = parser.parse_args()
	
	input_image_folder = args.input_img_folder
	output_image_folder = args.output_img_folder
	image_operations(input_image_folder, output_image_folder)








              