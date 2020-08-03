import argparse 
import os.path as path
import torch
import torchvision.transforms as transforms
from ModelGAN import Generator

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("output", type=str)
parser.add_argument("--num-images", "-n", type=int)
args = parser.parse_args()

G = Generator()
G.load_state_dict(torch.load(args.model))

for i in range(args.num_images):
    z = torch.randn(100)
    y = G(z).view(28, 28)
    y_image = transforms.ToPILImage()(y)
    y_image.save(path.join(args.output, "image%d.bmp" % i))
