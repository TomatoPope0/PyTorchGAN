import argparse
import torch
import torchvision
from ModelGAN import Generator, Discriminator

# Prevent recursive subprocess creation on Windows
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", "-n", type=int, default=1)
    args = parser.parse_args()

    batch_size = 100
    mnist_train = torchvision.datasets.MNIST(
        "../Datasets/MNIST_PyTorch/",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    mnist_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_threads
    )

    D = Discriminator()
    G = Generator()

    criterion = torch.nn.BCELoss()
    # TODO: Add lr decay
    optimizer_D = torch.optim.SGD(
        D.parameters(),
        lr=0.1,
        momentum=0.5,
    )
    optimizer_G = torch.optim.SGD(
        G.parameters(),
        lr=0.1,
        momentum=0.5,
    )

    report_rate = 50
    num_epochs = 10
    for epoch in range(num_epochs):
        d_loss = 0.0
        g_loss = 0.0
        for i, data in enumerate(mnist_loader):
            # Train D
            ## Train on real data
            optimizer_D.zero_grad()

            outputs = D(data[0].view(batch_size, 784))
            loss = criterion(outputs, torch.ones(batch_size, 1))
            loss.backward()
            optimizer_D.step()

            d_loss += loss.item()

            ## Train on fake data
            optimizer_D.zero_grad()

            z = torch.randn(batch_size, 100)
            outputs = D(G(z).detach())
            loss = criterion(outputs, torch.zeros(batch_size, 1))
            loss.backward()
            optimizer_D.step()

            d_loss += loss.item()

            # Train G
            optimizer_G.zero_grad()
            
            # Can I use same z without resampling?
            z = torch.randn(batch_size, 100)
            outputs = D(G(z))
            loss = criterion(outputs, torch.ones(batch_size, 1))
            loss.backward()
            optimizer_G.step()

            g_loss += loss.item()

            if i % report_rate == report_rate-1:
                print("E: %d - [D: %.7f G: %.7f]" % 
                    (epoch, d_loss / report_rate, g_loss / report_rate))
                d_loss = 0.0
                g_loss = 0.0

    torch.save(G.state_dict(), "./Model/G.pt")
