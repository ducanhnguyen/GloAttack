from src.models.YOLOv5Model import YOLOv5Model
from src.models.FasterRCNNModel import FasterRCNNModel
from src.models.DETRModel import DETRModel


class ModelFactory:
    """Factory class để tạo models"""

    _models = {
        'yolov5': YOLOv5Model,
        'faster_rcnn': FasterRCNNModel,
        'detr': DETRModel,
    }

    @classmethod
    def create_model(cls, model_name, device, **kwargs):
        """Tạo model theo tên"""
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not supported. Available: {list(cls._models.keys())}")

        model_class = cls._models[model_name]
        # ✅ Truyền tất cả kwargs, bao gồm cả model_name nếu có
        return model_class(device, **kwargs)

    @classmethod
    def register_model(cls, name, model_class):
        """Đăng ký model mới"""
        cls._models[name] = model_class

    @classmethod
    def list_models(cls):
        """Liệt kê các model có sẵn"""
        return list(cls._models.keys())

    # @classmethod
    # def get_model_variants(cls, model_name):
    #     """Lấy các variant có sẵn cho từng model"""
    #     variants = {
    #         'yolov5': ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'],
    #         'faster_rcnn': [
    #             'fasterrcnn_resnet50_fpn',
    #             'fasterrcnn_mobilenet_v3_large_fpn',
    #             'fasterrcnn_mobilenet_v3_large_320_fpn'
    #         ],
    #         'detr': [
    #             'facebook/detr-resnet-50',
    #             'facebook/detr-resnet-101',
    #             'facebook/detr-resnet-50-panoptic',
    #             'facebook/detr-resnet-101-panoptic'
    #         ]
    #     }
    #     return variants.get(model_name, [])