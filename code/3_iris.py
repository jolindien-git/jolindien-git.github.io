import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

FILE_NAME = "iris.data"


# lecture fichier
data_array = np.loadtxt(FILE_NAME, delimiter=",")
np.random.shuffle(data_array) # mélange les données

rows, cols = data_array.shape

print("taille des données = %i " % rows)

sepal_length = data_array[:, 0]
sepal_width = data_array[:, 1]
petal_length = data_array[:, 2]
petal_width = data_array[:, 3]
iris_class = data_array[:, 4]


#%% analyse des données

# visualisation
plt.subplot(231)
plt.plot(sepal_length, '.'); plt.xlabel("sepal_length")
plt.subplot(232)
plt.plot(sepal_width, '.'); plt.xlabel("sepal_width")
plt.subplot(233)
plt.plot(petal_length, '.'); plt.xlabel("petal_length")
plt.subplot(234)
plt.plot(petal_width, '.'); plt.xlabel("petal_width")
plt.subplot(235)
plt.plot(iris_class, '.'); plt.xlabel("iris_class")
plt.show()

# statistique simple + correlation avec iris_class
def stats(values, name):
    correlation = np.corrcoef(values, iris_class)[1, 0]
    print("%s %.1f \t %.1f \t %.2f \t %.2f \t %.4f" % 
          (name, np.min(values), np.max(values), np.mean(values), np.std(values), correlation))

print("\t\t\t Min  \t Max   \t Mean    \t SD   \t Class Correlation")
stats(sepal_length, "sepal length")
stats(sepal_width, "sepal width")
stats(petal_length, "petal length")
stats(petal_width, "petal width")


# matrice de corrélation
mat = np.stack([sepal_length, sepal_width, petal_length, petal_width, iris_class])
corr_matrix = np.corrcoef(mat)
nb = mat.shape[0]
print(corr_matrix)

fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap="coolwarm")
ax.xaxis.set(ticks=np.arange(nb), ticklabels=("sepal l", "sepal w", "petal l", "petal w", "iris class"))
ax.yaxis.set(ticks=np.arange(nb), ticklabels=("sepal length", "sepal width", "petal length", "petal width", "iris class"))
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
for i in range(nb):
    for j in range(nb):
        ax.text(j, i, "%.2f" % corr_matrix[i, j], ha='center', va='center', color='r')
plt.show()


#%% datasets
BATCH_SIZE = 32


class Iris_Dataset(Dataset):

    def __init__(self, train=True):
        super().__init__()
        
        # sélection données (train / valid)
        n_train = 8 * rows // 10 # 80% des données pour l'entrainement
        sl = slice(0, n_train) if train else slice(n_train, rows)
        self.sepal_length = sepal_length[sl]
        self.sepal_width = sepal_width[sl]
        self.petal_length = petal_length[sl]
        self.petal_width = petal_width[sl]
        self.iris_class = iris_class[sl]
        
        self.N = n_train if train else rows - n_train
        
        # normalisation des données numérique
        #       attention : les stats (mean et std) doivent être calculées sur l'ensemble des données
        self.sepal_length = (self.sepal_length - sepal_length.mean()) / sepal_length.std()
        self.sepal_length = torch.tensor(self.sepal_length, dtype=torch.float32).unsqueeze(1) # shape = (data_size, 1)
        self.sepal_width = (self.sepal_width - sepal_width.mean()) / sepal_width.std()
        self.sepal_width = torch.tensor(self.sepal_width, dtype=torch.float32).unsqueeze(1) # shape = (data_size, 1)
        self.petal_length = (self.petal_length - petal_length.mean()) / petal_length.std()
        self.petal_length = torch.tensor(self.petal_length, dtype=torch.float32).unsqueeze(1) # shape = (data_size, 1)
        self.petal_width = (self.petal_width - petal_width.mean()) / petal_width.std()
        self.petal_width = torch.tensor(self.petal_width, dtype=torch.float32).unsqueeze(1) # shape = (data_size, 1)
        
        self.x = torch.cat((self.sepal_length, self.sepal_width, self.petal_length, self.petal_width), dim = 1)        
        self.y = torch.tensor(self.iris_class, dtype=torch.int64)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.N


train_data = Iris_Dataset(train=True)
valid_data = Iris_Dataset(train=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))

x, y = train_data[0]
IN_SIZE = len(x)
OUT_SIZE = 3


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


