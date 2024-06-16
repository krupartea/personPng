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


# combine the input transforms with the model into a single Module
class DeepLab(nn.Module):

    def __init__(self):
        super().__init__()
        # self.transforms = transforms
        self.deeplab = deeplab

    def forward(self, x):
        # x = self.transforms(x)
        x = self.deeplab(x)
        x = x["out"]
        # 1.0 if the highest probability of a pixel corresponds to the `person` class
        # 0.0 otherwise
        x = 1 - (torch.argmax(x, 1) == 0).type(torch.FloatTensor)
        return x


model = DeepLab()
input = torch.randn(1, 3, 520, 520)
onnx_program = torch.onnx.export(model, input, "model.onnx")
