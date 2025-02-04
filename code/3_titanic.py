import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

FILE_NAME = "titanic.csv"


# lecture fichier
data_array = np.loadtxt(FILE_NAME, delimiter=",")

rows, cols = data_array.shape

print("taille des données = %i " % rows)

p_id = data_array[:, 0]
survived = data_array[:, 1]
cabin_class = data_array[:, 2]
female = data_array[:, 3]
fare = data_array[:, 4]


#%% analyse des données

# visualisation
plt.subplot(131)
plt.plot(p_id, '.'); plt.xlabel("p_id")
plt.subplot(132)
plt.plot(cabin_class, '.'); plt.xlabel("cabin_class")
plt.subplot(133)
plt.plot(fare, '.'); plt.xlabel("fare")
plt.show()

# statistique simple
print("fare: min %.3f  max % .3f écart-type  %.3f" %
      (fare.min(), fare.max(), fare.std()))


# correlation entre 2 grandeurs
x, y = p_id, survived
corr_matrix = np.corrcoef(x, y)
np.cov
print(corr_matrix)

def get_pearson_coeff(x, y):
    xm, ym = x.mean(), y.mean()
    return ((x-xm)*(y-ym)).sum() / ( np.sqrt(((x-xm)**2).sum()) * np.sqrt(((y-ym)**2).sum()))
    
print("pearson: %.2f" % get_pearson_coeff(x, y))
print("pearson: %.2f" % get_pearson_coeff(x, x))


# correlation entre plusieurs grandeurs
mat = np.stack([p_id, survived, cabin_class, female, fare])
corr_matrix = np.corrcoef(mat)
nb = mat.shape[0]
print(corr_matrix)

fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap="coolwarm")
ax.xaxis.set(ticks=np.arange(nb), ticklabels=("id", "survived", "cabin", "female", "tarif"))
ax.yaxis.set(ticks=np.arange(nb), ticklabels=("id", "survived", "cabin", "female", "tarif"))
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
for i in range(nb):
    for j in range(nb):
        ax.text(j, i, "%.2f" % corr_matrix[i, j], ha='center', va='center', color='r')
plt.show()


#%% datasets
BATCH_SIZE = 32

np.random.shuffle(data_array) # mélange les données

class Titanic_Dataset(Dataset):

    def __init__(self, train=True):
        super().__init__()
        
        # sélection données (train / valid)
        n_train = 8 * rows // 10 # 80% des données pour l'entrainement
        sl = slice(0, n_train) if train else slice(n_train, rows)
        self.p_id = p_id[sl]
        self.survived = survived[sl]
        self.cabin_class = cabin_class[sl]
        self.female = female[sl]
        self.fare = fare[sl]
        
        self.N = n_train if train else rows - n_train
        
        # normalisation des données numérique
        #       attention : les stats (mean et std) doivent être calculées sur l'ensemble des données
        self.fare = (self.fare - fare.mean()) / fare.std()
        self.fare = torch.tensor(self.fare, dtype=torch.float32).unsqueeze(1) # shape = (data_size, 1)
        
        # encodage one-hot avec torch
        def encode_one_hot(v):
            new_v = torch.tensor(v, dtype=torch.int64)
            new_v = F.one_hot(new_v)
            return new_v
        
        self.p_id = encode_one_hot(self.p_id)
        self.cabin_class = encode_one_hot(self.cabin_class)
        self.female = encode_one_hot(self.female)
        
        self.x = torch.cat((self.cabin_class, self.female, self.fare), dim = 1)        
        self.y = torch.tensor(self.survived, dtype=torch.int64)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.N


train_data = Titanic_Dataset(train=True)
valid_data = Titanic_Dataset(train=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

x, y = train_data[0]
IN_SIZE = len(x)
OUT_SIZE = 2 #len(y)


def get_score(y, y_predict):
    class_predict = y_predict.argmax(dim=1)
    nb_true = (y == class_predict).sum().item()
    nb_total = len(y)
    score_percent = nb_true / nb_total * 100
    return nb_true, nb_total, score_percent
    
def get_valid_loss(model, verbose=False):
    x, y = list(valid_loader)[0]
    with torch.no_grad():
        # cross entropy loss
        y_predict = model(x)
        valid_loss = F.cross_entropy(y_predict, y).item()
        # score (nombre de classe predictes)
        nb_true, nb_total, score_percent = get_score(y, y_predict)
        if verbose:
            print("loss : %.3e" % valid_loss)
            print("score : %i/%i = %.2f percent" %(nb_true, nb_total, score_percent) )
    return valid_loss, score_percent


#%% MLP

HIDDEN_SIZE = 32

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=torch.nn.Linear(IN_SIZE, HIDDEN_SIZE) # couche cachée
        # self.fc1_bis=torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) # couche cachée
        self.fc2=torch.nn.Linear(HIDDEN_SIZE, OUT_SIZE)

    def forward(self, x):
        y = self.fc1(x)
        y = torch.functional.F.relu(y)
        # y = self.fc1_bis(y)
        # y = torch.functional.F.relu(y)
        y = self.fc2(y)
        return y

model = MLP()

# test: loss avant entrainement
get_valid_loss(model, True)


#%% entrainement du MLP
LEARNING_RATE = 1e-3
EPOCHS = 500

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
train_losses, valid_losses = [], [] # for each epoch
train_scores, valid_scores = [], []
lr = LEARNING_RATE
for epoch in range(EPOCHS):
    # entrainement
    losses, scores = [], []
    for x, y in train_loader:
        # evaluer fonction cout
        y_predict = model(x)
        loss = F.cross_entropy(y_predict, y)
        losses.append(loss.item())
        scores.append(get_score(y, y_predict)[-1])

        # mise à jour paramètre
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(losses)
    train_losses.append(train_loss)
    train_score = np.mean(scores)
    train_scores.append(train_score)
    
    # vaidation
    valid_loss, valid_score = get_valid_loss(model)
    valid_losses.append(valid_loss)
    valid_scores.append(valid_score)
    
    # afficher progression
    if epoch % 10 == 0:
      print('epoch %i train %.2e %.2f  valid %.2e %.2f' % 
            (epoch, train_loss, train_score, valid_loss, valid_score))
     
    # if epoch > 0 and epoch % 100 == 0:
    #     lr /= 2
    #     print("decrease lr %.2e" % lr)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# visualiser progression loss (train vs valid)
plt.plot(train_scores)
plt.plot(valid_scores)
plt.legend(['train score', 'valid score'])
plt.xlabel('epoch')
plt.show()

plt.semilogy(train_losses)
plt.semilogy(valid_losses)
plt.legend(['train loss', 'valid loss'])
plt.xlabel('epoch')
plt.show()