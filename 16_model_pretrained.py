import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("data_image_net", split='train',
#                                            ransform=torchvision.transforms.ToTensor())

vgg16 = torchvision.models.vgg16(pretrained=True)

print(vgg16)

vgg16.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16)
