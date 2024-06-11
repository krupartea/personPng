from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
import torch.nn as nn
from torchvision.transforms import v2
import torch
from torchvision.transforms import InterpolationMode


deeplab = deeplabv3_mobilenet_v3_large(
    weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
)
deeplab.eval()


# the input must be transformed for the model to work correctly
# corresponding transformations are defined in `DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms`:
# SemanticSegmentation(
#     resize_size=[520]
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]
#     interpolation=InterpolationMode.BILINEAR
# )
# we can't directly export the to ONNX and use in C++ code

# therefore, recreate these transforms manually as Module and embed into the model's forward method
transforms = nn.Sequential(
            v2.Resize(520, InterpolationMode.NEAREST, antialias=True),  # NEAREST, since BILINEAR is not supported in my ONNX export
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )


# combine the input transforms with the model into a single Module
class DeepLab(nn.Module):

    def __init__(self):
        super().__init__()
        self.transforms = transforms
        self.deeplab = deeplab

    def forward(self, x):
        x = self.transforms(x)
        x = self.deeplab(x)
        x = x["out"][0, 0]  # first batch, first class map ("Person")
        return x


model = DeepLab()
input = torch.randn(1, 3, 32, 32)
onnx_program = torch.onnx.export(model, input, "model.onnx")
