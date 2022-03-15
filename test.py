from tqdm import tqdm
import torch
from dataset import get_loader
from model import FBSResMLP


model = FBSResMLP(384, 32, 12, 3, 4, 10)
model.cuda()
state_dict = torch.load(
    f'checkpoints/best_Lamb_fbs=False.pt')
model.load_state_dict(state_dict)
trainloader, valloader, testloader = get_loader(batch_size=256)
for epoch in range(50):
    with torch.no_grad():
        total_step = len(testloader)
        test_loss = 0
        total_num = 0
        correct_num = 0
        model.eval()
        for data, labels in tqdm(testloader, total=total_step):
            data = data.cuda()
            labels = labels.cuda()
            prediction = model(data)
            loss = torch.nn.functional.cross_entropy(prediction, labels)
            test_loss += loss.item()
            _, prediction = prediction.max(dim=1)
            total_num += labels.shape[0]
            correct_num += prediction.eq(labels).sum().item()
        test_loss = test_loss / total_step
        test_acc = 100. * correct_num / total_num
        print(
            f'Epoch {epoch} Test loss: {test_loss}--Test Acc: {test_acc}')
