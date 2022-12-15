# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/15 12:01
  @version V1.0
"""
import os
import shutil

# 重命名yolo的图片和对应的txt文件名，新名字都是从0开始的数字

input_path = './data'
output_path = './output'
# 判断文件夹是否存在
if not os.path.exists(output_path):
	os.makedirs(output_path)
if not os.path.exists(output_path + '/images'):
	os.makedirs(output_path + '/images')
if not os.path.exists(output_path + '/labels'):
	os.makedirs(output_path + '/labels')

# 移动classes.txt文件
if not os.path.exists(input_path + '/labels/classes.txt'):
	print('classes.txt文件不存在')
	exit()
shutil.copy(input_path + '/labels/classes.txt', output_path + '/labels/classes.txt')

# 读取文件夹下的所有文件
images = os.listdir(input_path + '/images')
labels = os.listdir(input_path + '/labels')

count = 0
for image in images:
	# 获取文件名，后缀
	image_name, image_suffix = os.path.splitext(image)
	new_image_name = str(count) + image_suffix
	new_label_name = str(count) + '.txt'
	# 复制图片和对应的txt文件
	shutil.copy(input_path + '/images/' + image, output_path + '/images/' + new_image_name)
	shutil.copy(input_path + '/labels/' + image_name + '.txt', output_path + '/labels/' + new_label_name)
	count += 1
print('共处理', count, '张图片')
