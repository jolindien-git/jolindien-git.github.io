import numpy as np

# lire le fichier .csv
file_name = "titanic_temp.csv"
d = np.loadtxt(file_name, dtype=np.str_, delimiter=",")
print("type:", type(d))
print("shape:", d.shape)
print(d)
print("field names:", d[0])

# traitement
keys = d[0] 
values = d[1:]
p_id = np.int32([np.arange(len(values))])
survived = np.int32([values[:, 0]])
cabin_class = np.int32([values[:, 1]])
female = np.int32([[v == "female" for v in values[:, 4]]])
fare = np.float32([values[:, 9]])
# embarked = np.str_([[]])
a = np.concatenate((p_id, survived, cabin_class, female, fare)).transpose()

# sauvegarde
np.savetxt("titanic.csv", a, fmt=["%i", "%i","%i", "%i", "%.3f"], 
           header="p_id, survived, cabin_class, female, fare", delimiter=",")
a = np.loadtxt("titanic.csv", delimiter=",")