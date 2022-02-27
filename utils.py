import torch
from math import ceil
from tqdm import tqdm

def global_avgpool2d(x):
    # input : a tensor with size [batch, C, H, W]
    x = torch.mean(torch.mean(x, dim=-1), dim=-1)
    return x  # [batch, C]

def winner_take_all(x, sparsity_ratio):
    # input : a tensor with size [batch, C]
    if sparsity_ratio < 1.0:
        k = ceil((1 - sparsity_ratio) * x.size(-1))
        inactive_idx = (-x).topk(k - 1, 1)[1]
        return x.scatter_(1, inactive_idx, 0)
    else:
        return x

def train(epochs, trainloader, testloader, costFunc, model, device, optimizer):
    best_acc = 0
    for epoch in range(epochs):
        train_loss = 0.
        total_num = 0
        correct_num = 0
        model.train()
        total_step = len(trainloader)
        for data, labels in tqdm(trainloader, total=total_step):
            data, labels = data.to(device), labels.to(device)
            prediction, lasso = model(data)
            loss = costFunc(prediction, labels) + lasso * 1e-8
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, prediction = prediction.max(dim=1)
            total_num += labels.shape[0]
            correct_num += prediction.eq(labels).sum().item()
        train_loss = train_loss / total_step
        train_acc = 100. * correct_num / total_num
        with torch.no_grad():
            total_step = len(testloader)
            test_loss = 0
            total_num = 0
            correct_num = 0
            model.eval()
            for data, labels in tqdm(testloader, total=total_step):
                data, labels = data.to(device), labels.to(device)
                prediction, lasso = model(data)
                loss = costFunc(prediction, labels) + lasso * 1e-8
                test_loss += loss.item()
                _, prediction = prediction.max(dim=1)
                total_num += labels.shape[0]
                correct_num += prediction.eq(labels).sum().item()
            test_loss = test_loss / total_step
            test_acc = 100. * correct_num / total_num
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       f'/best_fbs=True_0.5.pt')
        with open(f'/train_log_fbs=True_0.5.tsv', 'a') as log_file:
            log_file.write(
                f'{epoch}\t{train_loss}\t{test_loss}\t{train_acc}\t{test_acc}\t{best_acc}\n')
        print(
            f'/Epoch {epoch} Train Loss: {train_loss}--Train Acc: {train_acc} Test loss: {test_loss}--Test Acc: {test_acc}')