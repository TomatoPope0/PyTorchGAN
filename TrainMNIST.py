import argparse
import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--num_threads", "-n", type=int, default=1)
args = parser.parse_args()

mnist_train = torchvision.datasets.MNIST(
    "../Datasets/MNIST_PyTorch/",
    train=True,
    download=True
)
mnist_loader = torch.utils.data.DataLoader(
    mnist_train,
    batch_size=100,
    shuffle=True,
    num_workers=args.num_threads
)

