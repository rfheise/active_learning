from torch import nn
import torch 
from Cifar import get_data, DataSetFromTensor, get_data_cat_dog
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

device = "cuda"
class BalancedLoss(nn.Module):

    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes 
        self.device = device

    def forward(self, preds, y):
        preds = preds.to(self.device)
        y = y.to(self.device)
        weights = torch.unique(torch.cat((y,torch.arange(self.num_classes).to(self.device)),0), return_counts=True)[1].to(self.device)
        weights = 1/weights
        weights = weights/weights.sum()
        return nn.functional.cross_entropy(preds, y, weight=weights)

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(5,5),stride=(1,1)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(5,5),stride=(1,1)),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(4,4),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3),stride=(1,1)),
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(4,4),stride=(1,1)),
            nn.MaxPool2d(kernel_size=(2,2),stride=(1,1)),
            nn.Flatten(),
            nn.Linear(288, 256), 
            nn.ReLU(), 
            nn.Linear(256,64),
            nn.ReLU(), 
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, X):

        return self.layers(X)

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
        print(train(model, X_train, y_train, optimizer, loss))
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
        # transforms.Resize(36),
        # transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5),),
    ])

default_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,),),
    ])

def train(model, X,y, optimizer, criterion):

    model.train()
    rolling_acc = 0
    count = 0
    loader = DataLoader(DataSetFromTensor(X,y, default_transform), batch_size=1000, num_workers=12,shuffle=True)
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