from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

image_path = "hymenoptera_data/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 2, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()