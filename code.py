from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 8, stride=2, padding=4)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 198, 8, stride=2, padding=4)
        self.bn2 = nn.BatchNorm2d(198) 

        self.conv3 = nn.Conv2d(198, 198, 4, stride=2, padding=4)
        self.bn3 = nn.BatchNorm2d(198)      

        self.pool = nn.MaxPool2d(4, 2)

        self.fc1 = nn.Linear(198*4*4,512)
        self.bnfc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512,10)


    def forward(self, x):
        

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        print(x.size())
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #test
        x=F.relu(self.bn3(self.conv3(x)))
        print(x.size())
        x = self.pool(x)

        x = x.view(-1, 198 * 4 * 4)
        x = F.relu(self.bnfc1(self.fc1(x)))
        x = self.fc2(x)
        
        return x




net = Net()


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0, 0, 0), (1, 1, 1))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        print(inputs.size())

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')







