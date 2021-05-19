import os
import sys
import argparse
import numpy as np
import utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_PATH = os.path.join(ROOT_DIR, 'data/')

window = 9
step = 3
channel = 6
image_number = 'one' #one image, else multi-temporal image

if image_number == 'one':
	# one image
	utils.collect_npy_data(DATA_PATH, DATA_PATH,
				'ponceRGBNSS_S2_190108_aoi1.tif',
                            	'ponce_depth_aoi1_10m_crct.tif',
                            	window, step, channel)
else:
	# multi-temporal image
	fimg_list = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'fimg_list.txt'))]
	fdepth_list = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'fdepth_list.txt'))]
	
	output_folder = os.path.join(ROOT_DIR, 'data/parts/') 
	if not os.path.exists(output_folder):
	    os.mkdir(output_folder)
	
	for i in range(len(fimg_list)):
	    utils.collect_npy_data(DATA_PATH, output_folder, fimg_list[i], fdepth_list[i], window, step, channel)
