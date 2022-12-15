# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/15 16:04
  @version V1.0
"""
# 用yolov5检测出的结果转换成coco格式

import json
import os

import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# 读取文件夹下的所有文件
input_path = './output/images'
output_path = './output'
device = ''
weights = './weights/best.pt'
imgsz = 640
source = input_path
coco_json_save = output_path + '/detect_coco.json'

# 创建coco格式的预测结果
coco_json = []

# 判断文件夹是否存在
if not os.path.exists(output_path):
	os.makedirs(output_path)

device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
	model.half()  # to FP16

dataset = LoadImages(source, img_size=imgsz, stride=stride)

names = model.module.names if hasattr(model, 'module') else model.names

# Run inference
if device.type != 'cpu':
	model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

for path, img, im0s, vid_cap in dataset:
	# 获取图片名字
	image_name = os.path.basename(path).split('.')[0]

	img = torch.from_numpy(img).to(device)
	img = img.half() if half else img.float()  # uint8 to fp16/32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	# Inference
	pred = model(img, augment=False)[0]
	# Apply NMS
	pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

	# Process detections
	for i, det in enumerate(pred):  # detections per image
		if len(det):
			# Rescale boxes from img_size to im0 size
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

			# Write results
			for *xyxy, conf, cls in reversed(det):
				# 将检测结果保存到coco_json中
				coco_json.append({
					'image_id': int(image_name),
					'category_id': int(cls) + 1,
					'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])],
					'score': float(conf),
					'area': float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
				})

# 保存json文件
with open(os.path.join(coco_json_save), 'w') as f:
	# indent=2 保存json文件时，缩进2个空格
	json.dump(coco_json, f, indent=2)

print(len(coco_json), 'Done!')
