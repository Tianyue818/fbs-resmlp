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
    traindata = torchvision.datasets.CIFAR10(
        'data', train=True, transform=transform_train, download=True)
    testdata = torchvision.datasets.CIFAR10(
        'data', train=False, transform=transform_test, download=True)
    trainloader = DataLoader(traindata, batch_size=batch_size,
                              shuffle=True, pin_memory=True)
    testloader = DataLoader(testdata, batch_size=batch_size,
                             shuffle=False, pin_memory=True)
    return trainloader, testloader