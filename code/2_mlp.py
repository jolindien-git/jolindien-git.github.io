import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


#%% implementation MLP
M = 64 # taille couche cachée

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=torch.nn.Linear(1, M) # couche cachée
        self.fc1_bis=torch.nn.Linear(M, M) # couche cachée
        self.fc2=torch.nn.Linear(M, 1)

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
N_TRAIN = 200

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
    x = np.random.uniform(0, 1, size=(N, 1))
    y = np.sin(2 * np.pi * x) + np.random.normal(scale=.1, size=(N, 1))
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
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

valid_data = My_Dataset(50)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

# test
print("tailles : train %i  valid %i" % (len(train_data), len(valid_data))) # utilise __len___
print("échantillon : x, y = ", train_data[0]) # utilise __getitem__


#%% entrainement du MLP
LEARNING_RATE = 1e-1
EPOCHS = 2000
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    # entrainement
    train_losses = []
    for x, y in train_loader:
        # evaluer fonction cout
        y_predict = model(x)
        loss = ((y_predict - y)**2).mean()
        train_losses.append(loss.item())

        # mise à jour paramètre
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # vaidation
    with torch.no_grad(): # pas besoin de calcul de gradient
        for i, (x, y) in enumerate(valid_loader):
            assert(i == 0) # une seule boucle (pas de mini-batch)
            y_predict = model(x)
            valid_loss = ((y_predict - y)**2).mean().item()
    if epoch % 10 == 0:
      print('epoch %i train %.2e  valid %.2e' % (epoch, np.mean(train_losses),
                                                 valid_loss))

# visualiser résultat
plt.plot(x.numpy(), y.numpy(), 'o', x.numpy(), y_predict.numpy(), 'o')
plt.title('M=%i epochs=%i LR=%.2e' % (M, EPOCHS, LEARNING_RATE))
plt.legend(['data valid', 'MLP'])