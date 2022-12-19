# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/15 17:02
  @version V1.0
"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _eval(cocoGt, cocoDt, catIds, f, iouType='bbox'):
	"""
	param cocoGt: COCO ground truth
	param cocoDt: COCO detection results
	param catIds: array of category ids
	param iouType: 'bbox' or 'segm'
	"""
	cocoEval = COCOeval(cocoGt, cocoDt, iouType)
	cocoEval.params.catIds = catIds
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()
	f.write(str(cocoEval.stats) + '\n')


if __name__ == '__main__':
	pred_json = './output/detect_coco.json'
	anno_json = './output/gt_coco.json'

	# 使用COCO API加载预测结果和标注
	cocoGt = COCO(anno_json)
	cocoDt = cocoGt.loadRes(pred_json)

	# 获取所有gt的类别id
	catIds = cocoGt.getCatIds()

	with open('./output/coco_eval.txt', 'w') as f:
		for catId in catIds:
			# 获取类别名称
			catName = cocoGt.loadCats(catId)[0]['name']
			print('class name: ', catName)
			f.write('class name: ' + catName + '\n')
			_eval(cocoGt, cocoDt, [catId], f)
			print('----------------------------------------')
			f.write('----------------------------------------\n')

		# 获取所有类别的评估结果
		_eval(cocoGt, cocoDt, catIds, f)
