import torch
import torchvision.transforms as transforms
from ModelGAN import Generator

G = Generator()
G.load_state_dict(torch.load("./Model/G.pt"))

z = torch.randn(100)
y = G(z).view(28, 28)
y_image = transforms.ToPILImage()(y)
y_image.show()