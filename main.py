import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# DATASET
texts = [
    "i love this movie",
    "this film is amazing",
    "great acting and story",
    "i enjoyed the experience",
    "this is fantastic",
    "absolutely wonderful movie",
    "i hate this movie",
    "this film is terrible",
    "bad acting and boring story",
    "worst experience ever",
    "this is awful",
    "absolutely horrible movie"
]

labels = [1,1,1,1,1,1,0,0,0,0,0,0]


# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)


# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# TENSORS
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# DATALOADER
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# MODEL
class SentimentMLP(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, activation="relu"):

        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x

# TRAIN FUNCTION
def train_model(model, optimizer, epochs=50):

    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        train_losses.append(total_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

    return train_losses, train_accuracies

# EVALUATION FUNCTION
def evaluate_model(model):

    model.eval()

    with torch.no_grad():

        outputs = model(X_test)

        _, predicted = torch.max(outputs, 1)

        accuracy = accuracy_score(y_test, predicted)

        loss = loss_fn(outputs, y_test)

    return accuracy, loss.item()

# EXPERIMENT 1 — RELU
input_size = X_train.shape[1]

model_relu = SentimentMLP(
    input_size=input_size,
    hidden1=64,
    hidden2=32,
    activation="relu"
)

optimizer_relu = optim.Adam(model_relu.parameters(), lr=0.001)

relu_losses, relu_accuracies = train_model(model_relu, optimizer_relu)

relu_test_acc, relu_test_loss = evaluate_model(model_relu)

print("\nReLU Results")
print("Accuracy:", relu_test_acc)
print("Loss:", relu_test_loss)

# EXPERIMENT 2 — TANH
model_tanh = SentimentMLP(
    input_size=input_size,
    hidden1=64,
    hidden2=32,
    activation="tanh"
)

optimizer_tanh = optim.Adam(model_tanh.parameters(), lr=0.001)

tanh_losses, tanh_accuracies = train_model(model_tanh, optimizer_tanh)

tanh_test_acc, tanh_test_loss = evaluate_model(model_tanh)

print("\nTanh Results")
print("Accuracy:", tanh_test_acc)
print("Loss:", tanh_test_loss)

# VISUALIZATION
plt.plot(relu_losses, label="ReLU Loss")
plt.plot(tanh_losses, label="Tanh Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()


plt.plot(relu_accuracies, label="ReLU Accuracy")
plt.plot(tanh_accuracies, label="Tanh Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.show()

