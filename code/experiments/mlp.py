import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output(x)
        x = self.sigmoid(x)
        return x


def train_model(model, X_train, y_train, criterion, optimizer, num_epochs=10):
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')


def evaluate_model(model, X_test, y_test):
    
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        return accuracy.item()


def train_test_MLP(X_train, y_train, X_test, y_test):

    hidden_sizes = [300, 200, 100, 50]
    learning_rate = 0.001
    epochs = 10

    X_train = torch.stack(X_train)  
    y_train = torch.tensor(y_train).unsqueeze(1).float()

    X_test = torch.stack(X_test)
    y_test = torch.tensor(y_test)

    input_size = X_train.size()[1]
    output_size = y_train.size()[1]
    model = MultiLayerPerceptron(input_size, hidden_sizes, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, X_train, y_train, criterion, optimizer, num_epochs=epochs)
    accuracy = evaluate_model(model, X_test, y_test)
    
    return accuracy