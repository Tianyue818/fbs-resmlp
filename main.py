from model import FBSResMLP
from dataset import get_loader
from utils import train
import torch

def main():
  trainloader, testloader = get_loader(batch_size=256)
  model = FBSResMLP(768,
                32,
                3,
                3,
                16,
                10)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  costFunc = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  train(500,
        trainloader,
        testloader,
        costFunc,
        model,
        device,
        optimizer)

if __name__ == "__main__":
    main()
