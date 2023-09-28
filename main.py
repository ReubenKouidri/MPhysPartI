from my_utils.profiler import profiler
from src.CPSCDataset import CPSCDataset2D
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from models.defaults import S_2RB2D2


data_dir = "src/datasets/cpsc_data/test100"
ref_dir = "src/datasets/cpsc_data/reference300.csv"
RUNS = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_num_correct(preds, tgts):
    return preds.argmax(dim=1).eq(tgts).sum().item()


def train(model, trainloader, optimizer, criterion) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    for imgs, tgts in trainloader:
        optimizer.zero_grad()
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        preds = model(imgs)
        loss = criterion(preds, tgts)
        loss.backward()
        correct += get_num_correct(preds, tgts)
        total_loss += loss.item()
        optimizer.step()

    train_loss = total_loss / (len(trainloader.dataset) / trainloader.batch_size)
    train_acc = correct / (len(trainloader.dataset))

    return train_loss, train_acc


@profiler
def run(epochs):
    dataset = CPSCDataset2D(load_in_memory=True)
    dataloader = DataLoader(dataset, batch_size=50)
    model = S_2RB2D2()
    lr = 0.001
    optimiser = Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        a, b = train(model=model, trainloader=dataloader, optimizer=optimiser, criterion=criterion)


if __name__ == "__main__":
    run(RUNS)
