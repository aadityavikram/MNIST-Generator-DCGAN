from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def load_data(batch_size=32, size=64):
    train_data = datasets.MNIST(root='data/MNIST_data/',
                                transform=Compose([Resize(size),
                                                   ToTensor(),
                                                   Normalize(mean=[0.5], std=[0.5])
                                                   ]),
                                train=True,
                                download=False)

    test_data = datasets.MNIST(root='data/MNIST_data/',
                               transform=Compose([Resize(size),
                                                  ToTensor(),
                                                  Normalize(mean=[0.5], std=[0.5])
                                                  ]),
                               train=False,
                               download=False)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
