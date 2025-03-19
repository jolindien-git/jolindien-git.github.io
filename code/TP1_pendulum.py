import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


# %% dataset
SEQ_LEN_IN = 5
SEQ_LEN_OUT = 8

class Pendulum_Dataset(Dataset):
    g = 10.0
    dt = .05
    max_speed = 4 # speed = d(theta)/dt
    min_speed = -max_speed
    
    def __init__(self, sequences_number = 1, random_seed=0):
        np.random.seed(random_seed)
        
        # paramètres du pendule
        self.m = 1.0
        self.l = 1.0 # np.random.uniform(.5, 3, sequences_number)
        
        # conditions initiales
        theta = np.random.uniform(np.pi / 2, 3 * np.pi / 2, sequences_number)
        speed = np.random.uniform(self.min_speed, self.max_speed, sequences_number)
        
        # trajectoires
        sequences_length = SEQ_LEN_IN + SEQ_LEN_OUT
        self.thetas = np.zeros((sequences_number, sequences_length), dtype=np.float32)
        self.thetas[:, 0] = theta
        for k in range(1, sequences_length):
            theta, speed = self.step(theta.copy(), speed.copy())
            self.thetas[:, k] = theta
            
        # thetas: shape(sequences_number, sequences_length, 1)
        self.thetas.resize((sequences_number, sequences_length, 1))
        self.times = np.float32([k * self.dt for k in range(sequences_length)])

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
        return self.thetas.shape[0] # nombre de séquences

    def __getitem__(self, index):
        inputs = self.thetas[index, :SEQ_LEN_IN]
        outputs = self.thetas[index, SEQ_LEN_IN:]
        return self.times[:SEQ_LEN_IN], inputs, self.times[SEQ_LEN_IN:], outputs

# test dataset
dataset = Pendulum_Dataset(sequences_number=100, random_seed=42)
times_in, thetas_in, times_out, thetas_out = dataset[0]

#------------------- A FAIRE: tracer une trajectoire

#-------------------


# %% data loaders
N_SEQUENCES_TRAIN = 500
N_SEQUENCES_VALID = 300
N_SEQUENCES_TEST = 300

train_data = Pendulum_Dataset(sequences_number=N_SEQUENCES_TRAIN,
                              random_seed=0)
valid_data = Pendulum_Dataset(sequences_number=N_SEQUENCES_VALID,
                              random_seed=1)
test_data = Pendulum_Dataset(sequences_number=N_SEQUENCES_VALID,
                              random_seed=2)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))
test_loader = DataLoader(valid_data, batch_size=len(test_data))


# %% Elman

# hyperparamètres
#------------- A MODIFIER
HIDDEN_SIZE = 1
LEARNING_RATE = 1
EPOCHS = 2
#-------------

class Elman_RNN(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        # 1ère couche: entrée = entrée ET ancien état caché, sortie = nouvel état caché
        # (ici entrée = angle du pendule)
        self.i2h = torch.nn.Linear(1 + HIDDEN_SIZE, HIDDEN_SIZE)
        # 2eme couche: entrée = état caché, sortie = sortie du NN
        # (ici sortie = position du pendule)
        self.h2o = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x): # x: shape(batch_size, sequence_length, 1)
        batch_size = x.shape[0]
        assert x.shape[1] == SEQ_LEN_IN
        # init état caché
        hidden = torch.zeros(batch_size, HIDDEN_SIZE)
        # time-step loop on inputs
        for i in range(SEQ_LEN_IN):
            # concaténer entrée ET ancien état caché
            combined = torch.cat((x[:, i, :], hidden), 1) # shape(batch_size, HIDDEN_SIZE + 1)
            # nouvel état caché
            hidden = self.i2h(combined)
            hidden = torch.functional.F.tanh(hidden)
        # time-step loop on outputs
        y = torch.zeros([batch_size, SEQ_LEN_OUT, 1])
        y[:, 0, :] = self.h2o(hidden)
        for i in range(SEQ_LEN_OUT - 1):
            # concaténer entrée ET ancien état caché
            combined = torch.cat((y[:, i, :], hidden), 1) # shape(batch_size, HIDDEN_SIZE + 1)
            # nouvel état caché
            hidden = self.i2h(combined)
            hidden = torch.functional.F.tanh(hidden)
            y[:, i+1, :] = self.h2o(hidden)
        return y


model_RNN = Elman_RNN()


#%% training
optimizer = torch.optim.Adam(model_RNN.parameters(), lr=LEARNING_RATE)
train_losses, valid_losses = [], []
#---------------- A FAIRE: boucle d'entrainement

#---------------- 

# tracer les courbes de progression train/valid 
plt.figure()
plt.semilogy(train_losses)
plt.semilogy(valid_losses)
plt.legend(['train loss', 'valid loss'])
plt.xlabel("epochs")



#%% tracer une courbe prédite
#--------------- A FAIRE : 
#---------------


# %% MLP

# hyperparamètres
#------------- MODIFIER
HIDDEN_SIZE = 1
LEARNING_RATE = 1
EPOCHS = 2
#-------------

class MLP(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1=torch.nn.Linear(SEQ_LEN_IN, HIDDEN_SIZE) # couche cachée
        self.fcout=torch.nn.Linear(HIDDEN_SIZE, SEQ_LEN_OUT) # dernière couche

    def forward(self, x):# x: shape(batch_size, sequence_length, 1)
        batch_size = x.shape[0]
        assert x.shape[1] == SEQ_LEN_IN
        
        x_flat = x.flatten(start_dim=1)
        y = self.fc1(x_flat)
        y = torch.functional.F.relu(y)
        y = self.fcout(y)
        return y.unsqueeze(-1)

model_MLP = MLP()

# training
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=LEARNING_RATE)
train_losses, valid_losses = [], []
#---------------- COMPLETER

#---------------- 

# tracer les courbes de progression train/valid 
plt.semilogy(train_losses)
plt.semilogy(valid_losses)
plt.legend(['train loss', 'valid loss'])
plt.xlabel("epochs")

