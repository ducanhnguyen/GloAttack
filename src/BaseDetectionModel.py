from abc import ABC, abstractmethod
from io import BytesIO

import requests
from pycocotools.coco import COCO

from src.myutils import compute_map_per_image

from PIL import Image


class BaseDetectionModel(ABC):
    def __init__(self, device):
        self.device = device
        self.model = None
        self.names = []

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def preprocess_input(self, img_pil):
        pass

    @abstractmethod
    def compute_loss(self, img_tensor, targets):
        pass

    @abstractmethod
    def predict(self, img_tensor, conf_thresh=0.5):
        pass

    @abstractmethod
    def prepare_targets(self, anns, img_pil, cat_id_to_index, ratio=None, pad=None):
        pass

    def load_image_and_targets(self, img_id):
        coco_annotation_file = '../annotations/annotations/instances_val2017.json'
        data_dir = 'http://images.cocodataset.org/val2017/'
        coco = COCO(coco_annotation_file)

        img_info = coco.loadImgs(img_id)[0]
        img_url = data_dir + img_info['file_name']
        response = requests.get(img_url)
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            return None

        img_tensor = self.preprocess_input(img_pil)
        ori_tensor = img_tensor.clone().detach()
        targets, gt_boxes, gt_classes = self.prepare_targets(anns, img_pil)

        return img_tensor, ori_tensor, gt_boxes, gt_classes, targets

    def evaluate(self, ori_tensor, gt_classes, adv_tensor, gt_boxes):
        pred_boxes_orig, pred_scores_orig, pred_labels_orig = self.predict(ori_tensor)
        map_orig = compute_map_per_image(pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                                         gt_boxes, gt_classes, iou_thresh=0.5)

        pred_boxes_adv, pred_scores_adv, pred_labels_adv = self.predict(adv_tensor)
        map_adv = compute_map_per_image(pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                                        gt_boxes, gt_classes, iou_thresh=0.5)

        return (map_orig, map_adv, pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                pred_boxes_adv, pred_scores_adv, pred_labels_adv)

    def get_model_name(self):
        return self.__class__.__name__.lower().replace('model', '')

    def get_model_info(self):
        return {
            'name': self.get_model_name(),
            'device': self.device,
            'num_classes': len(self.names)
        }
