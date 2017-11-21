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

        
        self.conv1 = nn.Conv2d(1, 128, 8, padding=4)
        self.bn1 = nn.BatchNorm2d(128,eps=1e-3)

        self.conv2 = nn.Conv2d(128, 198, 8, padding=3)
        self.bn2 = nn.BatchNorm2d(198,eps=1e-3) 

        self.conv3 = nn.Conv2d(198, 198, 4, padding=3)
        self.bn3 = nn.BatchNorm2d(198,eps=1e-3)      

        self.pool = nn.MaxPool2d(4, 2)

        self.fc1 = nn.Linear(198*3*3,512)
        self.bnfc1 = nn.BatchNorm1d(512,eps=1e-3)

        self.fc2 = nn.Linear(512,10)


    def forward(self, x):
        
        #print(x.size())
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        #print(x.size())
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #print(x.size())
        #test
        x=F.relu(self.bn3(self.conv3(x)))
        #print("After 3rd conv and BN",x.size())
        x = self.pool(x)
        #print("After final pooling", x.size())

        x = x.view(-1, 198 * 3 * 3)

        x = F.relu(self.bnfc1(self.fc1(x)))
        x = self.fc2(x)
        
        return x




net = Net()


transform = transforms.Compose([transforms.Pad(2),transforms.ToTensor(),transforms.Normalize((0, 0, 0), (1, 1, 1))])


trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,  weight_decay=5e-4, nesterov=True)
optimizer2 = optim.SGD(net.parameters(), lr=2e-4, momentum=0.9,  weight_decay=5e-4, nesterov=True)


for epoch in range(60):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        #print(inputs.size())

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        optimizer2.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs.size())
        loss = criterion(outputs, labels)
        loss.backward()
        if epoch<30:
            optimizer.step()
        if epoch>29:
            optimizer2.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')







