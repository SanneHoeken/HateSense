import torch
import torch.nn as nn
import torch.optim as optim
#import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        in_features = input_size
        
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size

        self.out = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.out(x)
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
        _, predicted = torch.max(outputs.data, 1) 
        accuracy = (predicted == y_test).float().mean()
        return predicted.tolist(), accuracy.item()


def train_test_MLP(X_train, y_train, X_test, y_test, num_classes):
    
    torch.manual_seed(12)
    hidden_sizes = [300, 200, 100, 50]
    learning_rate = 0.001
    epochs = 10

    X_train = torch.stack(X_train)  
    y_train = nn.functional.one_hot(torch.tensor(y_train), num_classes=num_classes).float()

    X_test = torch.stack(X_test)
    y_test = torch.tensor(y_test)

    input_size = X_train.size()[1]
    model = MLP(input_size, hidden_sizes, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, X_train, y_train, criterion, optimizer, num_epochs=epochs)
    predictions, accuracy = evaluate_model(model, X_test, y_test)
    
    return predictions, accuracy