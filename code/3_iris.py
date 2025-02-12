import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

FILE_NAME = "iris.data"


# lecture fichier
data_array = np.loadtxt(FILE_NAME, delimiter=",")

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

