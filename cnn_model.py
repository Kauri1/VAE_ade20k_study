import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=3, input_size=128, num_classes=150, pooling=True):
        super(CNN, self).__init__()

        self.pooling = pooling
        self.num_classes = num_classes
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        if self.pooling:
            final_size = input_size // 8  # After 3 pooling layers
        else:
            final_size = input_size
            
        self.flatten_dim = 128 * final_size * final_size

        self.bn_flat = nn.BatchNorm1d(self.flatten_dim)

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.pooling:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))

        #x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        
        x = x.view(-1, self.flatten_dim)  # Flatten the tensor
        x = self.bn_flat(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN_1D(nn.Module):
    def __init__(self, in_channels=1, input_size=128, num_classes=150, pooling=True):
        super(CNN_1D, self).__init__()

        self.pooling = pooling
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        if self.pooling:
            final_size = input_size // 8  # After 3 pooling layers
        else:
            final_size = input_size

        self.flatten_dim = 128 * final_size

        self.bn_flat = nn.BatchNorm1d(self.flatten_dim)

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if self.pooling:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(-1, self.flatten_dim)  # Flatten the tensor
        x = self.bn_flat(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size=256, num_classes=150, hidden_sizes=[256, 256], dropout_rate=0.2):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x