# reference
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# Finetuning from a pretrained model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_fasterrcnn_resnet50_fpn(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model