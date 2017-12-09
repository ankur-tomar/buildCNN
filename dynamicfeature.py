from __future__ import division
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math


class Net(nn.Module):

    def __init__(self, feature_set):

        super(Net, self).__init__()

        self.feature_set = feature_set


        self.conv1 = nn.Conv2d(1, feature_set[0], 8, padding=4)
        self.bn1 = nn.BatchNorm2d(feature_set[0],eps=1e-3)

        self.conv2 = nn.Conv2d(feature_set[0], feature_set[1], 8, padding=3)
        self.bn2 = nn.BatchNorm2d(feature_set[1],eps=1e-3) 

        self.conv3 = nn.Conv2d(feature_set[1], feature_set[2], 5, padding=3)
        self.bn3 = nn.BatchNorm2d(feature_set[2],eps=1e-3)      

        self.pool = nn.MaxPool2d(4, 2)
        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(feature_set[2]*4*4,feature_set[3])
        self.bnfc1 = nn.BatchNorm1d(feature_set[3],eps=1e-3)

        self.fc2 = nn.Linear(feature_set[3],10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #print m.kernel_size[0], m.kernel_size[1], m.out_channels
            elif isinstance(m, nn.Linear):
                n = m.weight.size()[0] 
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x=F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(-1, self.feature_set[2] * 4 * 4)
        x = F.relu(self.bnfc1(self.fc1(x)))
        x = self.fc2(x)
        return x

def predict_y(X, net):

    class_pred = net(X)
    values, class_pred = torch.max(class_pred, 1)
    return class_pred

def getMetric(weights, t_not, t_not_l2, conv = True):
    

    if conv:
        t = getDiffFromMean(weights)
        #print t.size()
        t_l2 = getL2(weights)
        #print t_l2.size()

        a = torch.mul(t, t_not)

        num = torch.sum(a, 0)



    else:
        t = weights - torch.mean(weights)
        a = torch.mul(t, t_not)
        num = torch.sum(a,1)
        t_l2 = getL2(weights)
    den = t_l2 * t_not_l2

    #print num
    #print den

    c = (1 - (num/den))

    return c

def getDiffFromMean(weights):
    weights = weights.view(weights.size()[0], -1)
    means = torch.mean(weights, 1)
    w = weights.view(weights.size()[1], weights.size()[0])
    t = w - means
    return t

def getL2(weights):
    weights = weights.view(weights.size()[0], -1)
    t_l2 = torch.norm(weights, 2, 1)
    return t_l2



def main():
    feature_set = [1,1,1,1]
    net = Net(feature_set)
    weights = list(net.conv1.parameters())[0].data
    t_not = getDiffFromMean(weights)
    t_not_l2 = getL2(weights)

    weights = list(net.conv2.parameters())[0].data
    t_not_2 = getDiffFromMean(weights)
    t_not_l2_2 = getL2(weights)

    weights = list(net.conv3.parameters())[0].data
    t_not_3 = getDiffFromMean(weights)
    t_not_l2_3 = getL2(weights)

    weights = list(net.fc2.parameters())[0].data
    t_not_4 = getDiffFromMean(weights)
    t_not_l2_4 = getL2(weights)

    transform = transforms.Compose([transforms.Pad(2),transforms.ToTensor(),transforms.Normalize((0, 0, 0), (1, 1, 1))])


    trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,  weight_decay=5e-4, nesterov=True)
    optimizer2 = optim.SGD(net.parameters(), lr=2e-4, momentum=0.9,  weight_decay=5e-4, nesterov=True)

    num_of_features_to_add = 1


    eps = 1e-6

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


            weights = list(net.conv1.parameters())[0].data



            c1 = getMetric(weights, t_not, t_not_l2)

            a = torch.max(c1)

            print "Layer 1", a
            
            if a < 1 - eps:
                feature_set[0] += 1
                reset = True

            weights = list(net.conv2.parameters())[0].data

            c1 = getMetric(weights, t_not_2, t_not_l2_2)

            a = torch.max(c1)

            print "Layer 2", a
            
            if a < 1 - eps:
                feature_set[1] += 1
                reset = True

            weights = list(net.conv3.parameters())[0].data

            c1 = getMetric(weights, t_not_3, t_not_l2_3)

            a = torch.max(c1)

            print "Layer 3", a
            
            if a < 1 - eps:
                feature_set[2] += 1
                reset = True

            weights = list(net.fc2.parameters())[0].data

            c1 = getMetric(weights, t_not_4, t_not_l2_4, conv = False)

            a = torch.max(c1)

            print "Layer 4", a
            
            if a < 1 - eps:
                feature_set[3] += 1
                reset = True

            # weights = list(net.fc1.parameters())[0].data

            # c1 = getMetric(weights, t_not, t_not_l2)

            # a = torch.max(c1)
            
            # if a < 1 - eps:
            #     feature_set[0] += 1
            #     reset = True
            print feature_set




            # print statistics
            running_loss += loss.data[0]

            if i % 5 == 4:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            if reset:
                net = Net(feature_set)
                weights = list(net.conv1.parameters())[0].data
                t_not = getDiffFromMean(weights)
                t_not_l2 = getL2(weights)

                weights = list(net.conv2.parameters())[0].data
                t_not_2 = getDiffFromMean(weights)
                t_not_l2_2 = getL2(weights)

                weights = list(net.conv3.parameters())[0].data
                t_not_3 = getDiffFromMean(weights)
                t_not_l2_3 = getL2(weights)

                weights = list(net.fc2.parameters())[0].data
                t_not_4 = getDiffFromMean(weights)
                t_not_l2_4 = getL2(weights)

        torch.save(net, './saved_models/model_' + str(epoch))

    print('Finished Training')


if __name__ == '__main__':


    main()







