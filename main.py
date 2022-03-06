from model import FBSResMLP
from dataset import get_loader
from utils import train
from lamb import Lamb
import torch

def main():
  trainloader, valloader, testloader = get_loader(batch_size=256)
  model = FBSResMLP(384, 32, 6, 3, 4, 10)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
  costFunc = torch.nn.CrossEntropyLoss()
  optimizer = Lamb(model.parameters(), lr=0.005, weight_decay=0.1)

  train(300, trainloader, valloader, costFunc, model, device, optimizer)

if __name__ == "__main__":
    main()
