from torch import nn
import torch 
from Datasets.Cifar import get_data, DataSetFromTensor, get_data_cat_dog
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from Models.LeNet import LeNet,BalancedLoss

device = "cuda"


def main():

    epochs = 500
    model = LeNet().to(device)
    # model = models.resnet18()
    # model.fc = nn.Sequential(
    #     nn.Linear(model.fc.in_features, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 256),
    #     nn.ReLU(),
    #     nn.Linear(256, 10), 
    #     nn.Softmax(dim=1)
    # )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,250,500], gamma=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # loss = nn.CrossEntropyLoss()
    loss = BalancedLoss(10, device)

    X_train, y_train, X_test, y_test = get_data()

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    
    for i in range(epochs):
        print(f"\n\n\n-------------\nEpoch {i + 1}\n-------------\n\n\n")
        print("train accuracy: ",end="")
        print(train(model, X_train, y_train, optimizer, loss, scheduler))
        print("test accuracy: ",end="")
        print(test(model, X_test, y_test))

def get_entropy(proba):
    ent = -1 * proba * torch.log(proba)
    ent = torch.sum(ent, axis=1)
    return ent

train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomGrayscale(.1),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomVerticalFlip(.25),
        transforms.Resize(36),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5),),
    ])

default_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,),),
    ])

def train(model, X,y, optimizer, criterion, scheduler):

    model.train()
    rolling_acc = 0
    count = 0
    loader = DataLoader(DataSetFromTensor(X[:2000],y[:2000], default_transform), batch_size=50000, num_workers=12,shuffle=True)
    for X,y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step() 
        rolling_acc += (preds.argmax(dim=1) == y).sum().item()
        count += X.shape[0]
    scheduler.step()
    return rolling_acc/count
    
def test(model, X,y):

    model.eval()
    rolling_acc = 0
    count = 0
    pred_list = []
    loader = DataLoader(DataSetFromTensor(X,y, default_transform), batch_size=15000, num_workers=12,shuffle=True)
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            pred_list.append(preds)
            rolling_acc += (preds.argmax(dim=1) == y).sum().item()
            count += X.shape[0]

    return rolling_acc/count

if __name__ == "__main__":
    main()