import torch
import torchvision

mnist_train = torchvision.datasets.MNIST(
    "../Datasets/MNIST_PyTorch/",
    train=True,
    download=True
)
mnist_loader = torch.utils.data.DataLoader(
    mnist_train,
    batch_size=100,
    shuffle=True
)

