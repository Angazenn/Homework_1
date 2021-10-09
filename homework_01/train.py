import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def eval_acc(pred, true_label):
    pred_label = torch.argmax(pred, dim=1)
    return torch.sum(pred_label == true_label)


class ImgDataset(Dataset):

    def __init__(self, x, y, transform, device='cpu'):
        self.x = x
        self.y = torch.LongTensor(y)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.transform(self.x[index]).to(self.device), self.y[index].to(self.device)


class MY_MODULE(nn.Module):

    def __init__(self):
        super(MY_MODULE, self).__init__()
        # input [32, 32]
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # [32, 32]
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),  # [32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # [16, 16]

            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # [8, 8]

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),  # [8, 8]
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),  # [8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # [4, 4]

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 10, bias=True),
        )

    def forward(self, img):
        tmp_data = self.layers(img)
        output = self.fc(tmp_data.reshape(tmp_data.size(0), -1))
        return output


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

BATCH_SIZE = 64
LR = 0.005
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9
NUM_EPOCH = 30

train_data = []
train_label = []
data_dir = 'data/cifar_10'
for i in range(5):
    batch_data = unpickle(data_dir + '/data_batch_' + str(i + 1))
    train_data.append(batch_data[b'data'].reshape(-1, 32, 32, 3))
    train_label.append(np.array(batch_data[b'labels']))
train_data = np.concatenate(train_data, axis=0)
train_label = np.concatenate(train_label, axis=0)

batch_data = unpickle('data/cifar_10/test_batch')
test_data = batch_data[b'data'].reshape(-1, 32, 32, 3)
test_label = np.array(batch_data[b'labels'])

train_set = ImgDataset(train_data, train_label, train_transform, 'cuda')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = ImgDataset(test_data, test_label, test_transform, 'cuda')
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = MY_MODULE().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

for epoch in range(NUM_EPOCH):
    model.train()
    train_cor = 0
    train_tot = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data[0])
        loss = loss_fn(output, data[1])
        loss.backward()
        optimizer.step()
        train_cor += eval_acc(output, data[1])
        train_tot += len(data[0])
        if i % 50 == 49:
            print('epoch:{:d}, step:{:d}, loss={:.4f}'.format(epoch, i + 1, loss), end='\r')

    model.eval()
    test_cor = 0
    test_tot = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            output = model(data[0])
            test_cor += eval_acc(output, data[1])
            test_tot += len(data[0])
    print('epoch {:d}/{:d}, loss={:.4f}, train_acc={:.2f}, test_acc={:.2f}'.format(epoch, NUM_EPOCH, loss,
                                                                                   train_cor / train_tot * 100,
                                                                                   test_cor / test_tot * 100))