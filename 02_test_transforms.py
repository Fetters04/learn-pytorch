from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
