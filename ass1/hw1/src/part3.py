#!/usr/bin/env python3
"""
part3.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Linear(nn.Module):
    """
    DO NOT MODIFY
    Linear (10) -> ReLU -> LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # make sure inputs are flattened
        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)  # preserve batch dim

        return x


class FeedForward(nn.Module):
    """
    TODO: Implement the following network structure
    Linear (256) -> ReLU -> Linear(64) -> ReLU -> Linear(10) -> ReLU-> LogSoftmax
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)
        return x


class CNN(nn.Module):
    """
    TODO: Implement CNN Network structure

    conv1 (channels = 10, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    conv2 (channels = 50, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    Linear (256) -> Relu -> Linear (10) -> LogSoftmax


    Hint: You will need to reshape outputs from the last conv layer prior to feeding them into
    the linear layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(10, 50, 5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # x.shape = torch.Size([64, 1, 28, 28])
        # why does c1 take a 4 dimensional input only?
        x = self.conv1(x)
        # returns torch.Size([64, 10, 24, 24])
        # 28 + 1 - 5 = 24
        # print(f"shape after first convulutional layer = {x.shape}")
        x = F.relu(x)
        # print(f"shape after relu {x.shape}")
        x = self.maxpool(x)
        # print(f"shape after maxpool = {x.shape}")
        x = self.conv2(x)
        # print(f"shape after second convulutional layer = {x.shape}")
        x = F.relu(x)
        # print(f"shape after second relu {x.shape}")
        x = self.maxpool(x)
        # print(f"shape after second maxpool = {x.shape}")
        x = x.view(x.shape[0], -1)
        # print(f"reshape x before feeding to linear = {x.shape}")
        x = self.fc1(x)
        # print(f"shape after first linear {x.shape}")
        x = F.relu(x)
        # print(f"shape after relu {x.shape}")
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class NNModel:
    def __init__(self, network, learning_rate):
        """
        Load Data, initialize a given network structure and set learning rate
        DO NOT MODIFY
        """

        # Define a transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training data
        trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

        # Download and load the test data
        testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        self.model = network

        """
        TODO: Set appropriate loss function such that learning is equivalent to minimizing the
        cross entropy loss. Note that we are outputting log-softmax values from our networks,
        not raw softmax values, so just using torch.nn.CrossEntropyLoss is incorrect.
        
        Hint: All networks output log-softmax values (i.e. log probabilities or.. likelihoods.). 
        """
        # nn.CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        # since F.log_softmax is applied to the prediction of the models is this the correct 
        # function to use?
        self.lossfn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_train_samples = len(self.trainloader)
        self.num_test_samples = len(self.testloader)

    def view_batch(self):
        """
        TODO: Display first batch of images from trainloader in 8x8 grid

        Do not make calls to plt.imshow() here

        Return:
           1) A float32 numpy array (of dim [28*8, 28*8]), containing a tiling of the batch images,
           place the first 8 images on the first row, the second 8 on the second row, and so on

           2) An int 8x8 numpy array of labels corresponding to this tiling
        """
        first_batch = next(iter(self.trainloader))
        output_images = np.zeros([224, 224])
        output_labels = np.zeros([8, 8])
        images = first_batch[0] # torch.Size([64, 1, 28, 28])
        labels = first_batch[1] # torch.Size([64])
        row = 0
        column = 0
        for i in range(0, 64):
            image = images[i][0]
            output_labels[row][column] = labels[i] 
            if column == 0:
                y = 0
            else:
                y = (28 * column)
            if row == 0:
                x = 0
            else:
                x = (28 * row)
            print(f"{x}:{x+28}, {y}:{y+28}")
            output_images[x:x+28,y:y+28] = image[0:28, 0:28]
            column += 1
            if column == 8:
                row += 1
                column = 0
        return (output_images, output_labels)
        

    def train_step(self):
        """
        Used for submission tests and may be usefull for debugging
        DO NOT MODIFY
        """
        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return

    def train_epoch(self):
        self.model.train()
        for images, labels in self.trainloader:
            log_ps = self.model(images)
            loss = self.lossfn(log_ps, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def eval(self):
        self.model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                log_ps = self.model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        return accuracy / self.num_test_samples


def plot_result(results, names):
    """
    Take a 2D list/array, where row is accuracy at each epoch of training for given model, and
    names of each model, and display training curves
    """
    for i, r in enumerate(results):
        plt.plot(range(len(r)), r, label=names[i])
    plt.legend()
    plt.title("KMNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("./part_2_plot.png")


def main():
    models = [Linear(), FeedForward(), CNN()]  # Change during development
    epochs = 10
    results = []

    # Can comment the below out during development
    images, labels = NNModel(Linear(), 0.003).view_batch()
    print(labels)
    plt.imshow(images, cmap="Greys")
    plt.show()

    for model in models:
        print(f"Training {model.__class__.__name__}...")
        m = NNModel(model, 0.003)

        accuracies = [0]
        for e in range(epochs):
            m.train_epoch()
            accuracy = m.eval()
            print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")
            accuracies.append(accuracy)
        results.append(accuracies)

    plot_result(results, [m.__class__.__name__ for m in models])


if __name__ == "__main__":
    main()
