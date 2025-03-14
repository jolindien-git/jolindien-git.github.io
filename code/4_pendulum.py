import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

SEQUENCES_LENGTH = 40

class Pendulum_Dataset(Dataset):
    g = 10.0
    dt = .05
    max_speed = 4 # speed = d(theta)/dt
    min_speed = -max_speed
    
    def __init__(self, sequences_number = 1, sequences_length = 40):
        # paramètres du pendule
        self.m = 1.0
        self.l = np.random.uniform(.5, 3, sequences_number)
        
        # conditions initiales
        theta = np.random.uniform(np.pi / 2, 3 * np.pi / 2, sequences_number)
        speed = np.random.uniform(self.min_speed, self.max_speed, sequences_number)
        
        # trajectoire
        self.thetas = np.zeros((sequences_number, sequences_length), dtype=np.float32)
        self.thetas[:, 0] = theta
        for k in range(1, sequences_length):
            theta, speed = self.step(theta.copy(), speed.copy())
            self.thetas[:, k] = theta
            
        # thetas: shape(sequences_number, sequences_length, 1)
        self.thetas.resize((sequences_number, sequences_length, 1))

    def step(self, theta, thdot):
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = 0
        newthdot = thdot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = theta + newthdot * dt

        return newth, newthdot

    def __len__(self):
      return self.thetas.shape[0]

    def __getitem__(self, index):
      return self.thetas[index, :], self.l[index]


# test dataset
pendulum = Pendulum_Dataset(2, SEQUENCES_LENGTH)

thetas, l = pendulum[0]
plt.plot(thetas, "g*-")
plt.title('l = %.3f' % l)
plt.show()
thetas, l = pendulum[1]
plt.plot(thetas, "r*-")
plt.title('l = %.3f' % l)
plt.show()


#%% data loaders
N_SEQUENCES_TRAIN = 200
N_SEQUENCES_VALID = 30

train_data = Pendulum_Dataset(N_SEQUENCES_TRAIN, SEQUENCES_LENGTH)
valid_data = Pendulum_Dataset(N_SEQUENCES_VALID, SEQUENCES_LENGTH)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

def get_valid_loss(model, verbose=False):
    with torch.no_grad():
        thetas, l = list(valid_loader)[0]
        l_predict = model(thetas)
        valid_loss = ((l_predict - l)**2).mean().item()
    if verbose:
        print("===== validation (value by value): =====")
        print("\n".join("truth %.3f  predict %.3f" % (t,p) 
                        for t,p in zip(l, l_predict)))
    return valid_loss


#%% MLP
LEARNING_RATE = 1e-3
EPOCHS = 200

class MLP(torch.nn.Module):
    HIDDEN_SIZES = [100, 100] # paramètres (taille des couches cachées)
    
    def __init__(self):
        super().__init__()
        self.fc1=torch.nn.Linear(SEQUENCES_LENGTH, self.HIDDEN_SIZES[0]) # couche cachée
        self.fc2=torch.nn.Linear(self.HIDDEN_SIZES[0], self.HIDDEN_SIZES[1]) # couche cachée
        self.fc3=torch.nn.Linear(self.HIDDEN_SIZES[1], 1) # dernière couche

    def forward(self, x):
        x_flat = x.flatten(start_dim=1)
        y = self.fc1(x_flat)
        y = torch.functional.F.relu(y)
        y = self.fc2(y)
        y = torch.functional.F.relu(y)
        y = self.fc3(y)
        return y

model_MLP = MLP()

# initial valid loss
print("valid loss init:", get_valid_loss(model_MLP))

# training
optimizer = torch.optim.SGD(model_MLP.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    train_losses = []
    for x, y in train_loader:
        # evaluer fonction cout
        y_predict = model_MLP(x)
        loss = ((y_predict - y)**2).mean()
        train_losses.append(loss.item())

        # mise à jour paramètre
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # vaidation
    valid_loss = get_valid_loss(model_MLP)
    if epoch < 10 or epoch % 10 == 0:
      print('epoch %i train %.2e  valid %.2e' % (epoch, np.mean(train_losses),
                                                 valid_loss))

# analyze result
get_valid_loss(model_MLP, True)


#%% Vanilla Recurent networks

class Vanilla_RNN(torch.nn.Module):
    HIDDEN_SIZE = 100 # paramètre
    
    def __init__(self):
        super().__init__()
        # 1ère couche: entrée = entrée ET ancien état caché, sortie = nouvel état caché
        # (ici entrée = angle du pendule)
        self.i2h = torch.nn.Linear(1 + self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        # 2eme couche: entrée = état caché, sortie = sortie du NN
        # (ici sortie = longueur du pendule)
        self.h2o = torch.nn.Linear(self.HIDDEN_SIZE, 1)

    def forward(self, x): # x: shape(batch_size, SEQUENCES_LENGTH, 1)
        batch_size = x.shape[0]
        assert x.shape[1] == SEQUENCES_LENGTH
        assert x.shape[2] == 1
        # init état caché
        hidden = torch.zeros(batch_size, self.HIDDEN_SIZE)
        for i in range(SEQUENCES_LENGTH):
            # combiner entrée ET ancien état caché
            combined = torch.cat((x[:, i, :], hidden), 1) # shape(batch_size, HIDDEN_SIZE + 1)
            # nouvel état caché
            hidden = self.i2h(combined)
            hidden = torch.functional.F.tanh(hidden)
        y = self.h2o(hidden)
        return y


model_RNN = Vanilla_RNN()

# initial valid loss
print("valid loss init:", get_valid_loss(model_RNN))

# training
optimizer = torch.optim.SGD(model_RNN.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    train_losses = []
    for x, y in train_loader:
        # evaluer fonction cout
        y_predict = model_RNN(x)
        loss = ((y_predict - y)**2).mean()
        train_losses.append(loss.item())

        # mise à jour paramètre
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # vaidation
    valid_loss = get_valid_loss(model_RNN)
    if epoch < 10 or epoch % 10 == 0:
      print('epoch %i train %.2e  valid %.2e' % (epoch, np.mean(train_losses),
                                                 valid_loss))

# analyze result
get_valid_loss(model_RNN, True)