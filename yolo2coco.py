# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/15 16:44
  @version V1.0
"""

import datetime
import json
import os
import cv2

# 将yolo格式的数据集转换成coco格式的数据集

# 读取文件夹下的所有文件
images_path = './output/images'
labels_path = './output/labels'
output_path = './output'
coco_json_save = output_path + '/gt_coco.json'

# 创建coco格式的json文件
coco_json = {
	'info': {
		"description": "mpj Dataset",
		"url": "www.mpj520.com",
		"version": "1.0",
		"year": 2022,
		"contributor": "mpj",
		"date_created": datetime.datetime.utcnow().isoformat(' ')
	},
	"licenses": [
		{
			"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
			"id": 1,
			"name": "Attribution-NonCommercial-ShareAlike License"
		}
	],
	'images': [],
	'annotations': [],
	'categories': []
}

# 判断文件夹是否存在
if not os.path.exists(output_path):
	os.makedirs(output_path)
# 判断classes.txt文件是否存在
if not os.path.exists(labels_path + '/classes.txt'):
	print('classes.txt文件不存在')
	exit()

# 读取classes.txt文件
classes = []
with open(labels_path + '/classes.txt', 'r') as f:
	classes = f.readlines()
	classes = [c.strip() for c in classes]

# 创建coco格式的json文件
for i, c in enumerate(classes):
	coco_json['categories'].append({'id': i + 1, 'name': c, 'supercategory': c})

# 读取images文件夹下的所有文件
images = os.listdir(images_path)
for image in images:
	# 获取图片名和后缀
	image_name, image_suffix = os.path.splitext(image)
	# 获取图片的宽和高
	image_path = images_path + '/' + image
	img = cv2.imread(image_path)
	height, width, _ = img.shape
	# 添加图片信息
	coco_json['images'].append({
		'id': int(image_name),
		'file_name': image,
		'width': width,
		'height': height,
		'date_captured': datetime.datetime.utcnow().isoformat(' '),
		'license': 1
	})
	# 读取图片对应的标签文件
	label_path = labels_path + '/' + image_name + '.txt'
	if not os.path.exists(label_path):
		continue
	with open(label_path, 'r') as f:
		labels = f.readlines()
		labels = [l.strip() for l in labels]
		for j, label in enumerate(labels):
			label = label.split(' ')
			# 获取类别id
			category_id = int(label[0])
			# 将yolo格式的数据转换成coco格式的数据
			x = float(label[1]) * width
			y = float(label[2]) * height
			w = float(label[3]) * width
			h = float(label[4]) * height
			xmin = x - w / 2
			ymin = y - h / 2
			xmax = x + w / 2
			ymax = y + h / 2
			# 添加bbox信息
			coco_json['annotations'].append({
				'image_id': int(image_name),
				'category_id': category_id + 1,
				'bbox': [xmin, ymin, w, h],
				'id': len(coco_json['annotations']),
				'area': w * h,
				'iscrowd': 0,
				'segmentation': [],
				'attributes': ""
			})

# 保存json文件
with open(coco_json_save, 'w') as f:
	json.dump(coco_json, f, indent=2)

print(len(coco_json['images']), len(coco_json['annotations']), len(coco_json['categories']), 'Done!')
