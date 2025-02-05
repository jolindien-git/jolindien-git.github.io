import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


#%% implementation MLP
M = 100 # taille couche cachée

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=torch.nn.Linear(2, M) # couche cachée
        self.fc1_bis=torch.nn.Linear(M, M) # couche cachée
        self.fc2=torch.nn.Linear(M, 2)

    def forward(self, x):
        y = self.fc1(x)
        y = torch.functional.F.relu(y)
        y = self.fc1_bis(y)
        y = torch.functional.F.relu(y)
        y = self.fc2(y)
        return y

model = MLP()

# test
print(model)
print(list(model.parameters()))


#%% implémenter data sets
N_TRAIN = 2000
N_VALID = 500
BATCH_SIZE = 100

class My_Dataset(Dataset):
  '''
  Hérite de l'objet Dataset qui facilite la gestion des données.
  data:
        data_x: shape = (N, 1)
        data_y: shape = (N, 1)
        N est le nombre de données
        x et y sont scalaires
  '''
  def __init__(self, N = 100):
    # génerer données
    x = np.random.uniform(low=[-1, 1], high=[1, 4], size=(N, 2))
    y = np.zeros((N,2))
    x1, x2 = x[:, 0], x[:, 1]
    y[:, 0] = np.sqrt(x1 + x2)
    y[:, 1] = x2**3
    # mémoriser (le MLP nécessite des float sur 32 bits)
    self.data_x = np.float32(x)
    self.data_y = np.float32(y)
    self.N = N

  def __len__(self): # len = length
    return self.N

  def __getitem__(self, index):
    # retourne un couple (x, y)
    return self.data_x[index, :], self.data_y[index, :]

# train set, with mini-batch
train_data = My_Dataset(N_TRAIN)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

valid_data = My_Dataset(N_VALID)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

# test
print("tailles : train %i  valid %i" % (len(train_data), len(valid_data))) # utilise __len___
print("échantillon : x, y = ", train_data[0]) # utilise __getitem__


#%% entrainement du MLP
LEARNING_RATE = 1e-3
EPOCHS = 500
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

train_losses, valid_losses = [], [] # for each epoch
for epoch in range(EPOCHS):
    # entrainement
    losses = []
    for x, y in train_loader:
        # evaluer fonction cout
        y_predict = model(x)
        loss = ((y_predict - y)**2).mean()
        losses.append(loss.item())

        # mise à jour paramètre
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(losses)
    train_losses.append(train_loss)
    
    # vaidation
    with torch.no_grad(): # pas besoin de calcul de gradient
        x, y = list(valid_loader)[0]
        y_predict = model(x)
        valid_loss = ((y_predict - y)**2).mean().item()
        valid_losses.append(valid_loss)
    if epoch % 10 == 0:
      print('epoch %i train %.2e  valid %.2e' % (epoch, train_loss, valid_loss))
    if epoch == EPOCHS // 4:
        print("decrease lr")
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE / 4)


#%% visualiser progression loss (train vs valid)
plt.plot(train_losses)
plt.plot(valid_losses)
plt.legend(['train loss', 'valid loss'])
plt.xlabel('epoch')
plt.show()

plt.semilogy(train_losses)
plt.semilogy(valid_losses)
plt.legend(['train loss', 'valid loss'])
plt.xlabel('epoch')
plt.show()