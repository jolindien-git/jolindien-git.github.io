import numpy as np
from matplotlib import pyplot as plt

# lecture fichier
a = np.loadtxt("titanic.csv", delimiter=",")

p_id = a[:, 0]
survived = a[:, 1]
cabin_class = a[:, 2]
female = a[:, 3]
fare = a[:, 4]


#%% correlation entre 2 grandeurs

x, y = cabin_class, survived
corr_matrix = np.corrcoef(x, y)
xm, ym = x.mean(), y.mean()
R = ((x-xm)*(y-ym)).sum() / ( np.sqrt(((x-xm)**2).sum()) * np.sqrt(((y-ym)**2).sum()))
print(corr_matrix)
print("pearson:", R)


#%% correlation entre plusieurs grandeurs

mat = np.stack([p_id, survived, cabin_class, female, fare])
corr_matrix = np.corrcoef(mat)
nb = mat.shape[0]
corr_matrix

# figure
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap="coolwarm")
ax.xaxis.set(ticks=np.arange(nb), ticklabels=("id", "survived", "class", "female", "tarif"))
ax.yaxis.set(ticks=np.arange(nb), ticklabels=("id", "survived", "class", "female", "tarif"))
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
for i in range(nb):
    for j in range(nb):
        ax.text(j, i, "%.2f" % corr_matrix[i, j], ha='center', va='center', color='r')
plt.show()