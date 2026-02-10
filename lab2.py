import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

accuracy_list = []
# Data
X = torch.tensor([[i] for i in range(11)], dtype=torch.float32)
y = torch.tensor([[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]], dtype=torch.float32)

# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 3)     # 1 input neuron
        self.fc2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x)) # hidden layer sigmoid
        x = self.sigmoid(self.fc2(x)) # output layer sigmoid
        return x

model = MLP()

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(3000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    predictions = (y_pred > 0.5).float()
    accuracy = (predictions == y).float().mean()
    accuracy_list.append(accuracy.item())


    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item()}")

plt.plot(accuracy_list)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Epoch")
plt.show()

test = torch.tensor([[4.0], [6.0], [9.0]])
outputs = model(test)

print("Probabilities:\n", outputs.detach().numpy())
print("Predictions:\n", (outputs > 0.5).int())
