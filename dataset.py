import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
def get_loader(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data = torchvision.datasets.CIFAR10(
        'data', train=True, transform=transform_train, download=True)
    test_data = torchvision.datasets.CIFAR10(
        'data', train=False, transform=transform_test, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, pin_memory=True)
    return train_loader, test_loader