import torch
from torch import nn
import torch.nn.functional as F

class CategorizationModel(nn.Module):
    '''
    Basic example model using TF-IDF

    '''
    def __init__(self, input_dim, num_genres):
        super(CategorizationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.output = nn.Linear(64, num_genres)  # Output layer
        self.dropout = nn.Dropout(p=0.3)  # Dropout for regularization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = F.relu(x)  # Apply ReLU activation
        x = self.output(x)  # Output logits
        x = F.sigmoid(x) # Sigmoid activation for multi-label probabilities
        return x  
