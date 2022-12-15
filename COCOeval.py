# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/15 17:02
  @version V1.0
"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
	pred_json = './output/detect_coco.json'
	anno_json = './output/gt_coco.json'

	# 使用COCO API加载预测结果和标注
	cocoGt = COCO(anno_json)
	cocoDt = cocoGt.loadRes(pred_json)

	# 创建COCOeval对象
	cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

	# 执行评估
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

	# 保存结果
	with open('./output/coco_eval.txt', 'w') as f:
		f.write(str(cocoEval.stats))

	# 打印结果
	print(cocoEval.stats)
