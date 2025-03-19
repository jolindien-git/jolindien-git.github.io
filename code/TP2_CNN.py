import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np


# %% dataset : CIFAR10 (télécharge les images dans le dossier data)

transform = transforms.Compose([transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split train set into a train and validation sets
train_size = int(0.75*len(train_data))
valid_size = len(train_data) - train_size
torch.manual_seed(0)
train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

# nom des classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# %% comprendre les données

def print_image(image, title="classe de l'image ?"):
    npimg = image.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


# -- visualiser la 1ère image de train_data
image, label = train_data[0]
print_image(image)
print("=== image est de type :", type(image))
print("=== taille du tenseur :", image.shape)
print("=== label", label)


#------------------- A FAIRE: visualiser les 5 premières images AVEC le nom de sa classe
for i in range(1, 5):
    image, label = train_data[i]
    title = classes[label]
    print_image(image, title)
#-------------------



# %% modèle CNN

class LeNet(torch.nn.Module):
    def __init__(self):
        super (LeNet , self).__init__()
        # 3 input channels , 10 output channels,
        # 5x5 filters , stride=1, no padding
        self.conv1 = torch.nn.Conv2d(3, 20, 5, 1, 0)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1, 0)
        self.fc1 = torch.nn.Linear(5*5*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self , x):
        x = F.relu(self.conv1(x))
        # Max pooling with a filter size of 2x2 and a stride of 2
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = LeNet()

#------------- A FAIRE: comprendre max_pool2d
KERNEL = 2
STRIDE = 2
image, label = train_data[3]
print_image(image, classes[label])
image = F.max_pool2d(image, KERNEL, STRIDE)
print_image(image, "max_pool :kernel %i stride %i" % (KERNEL, STRIDE))
#-------------


# %% data loaders
BATCH_SIZE = 16
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data))
test_loader = DataLoader(valid_data, batch_size=len(test_data))


# %% entrainement

# hyperparamètres
#------------- A MODIFIER
LEARNING_RATE = 1e-3
EPOCHS = 20
#-------------


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses, valid_losses = [], [] # for each epoch
for epoch in range(EPOCHS):
    # entrainement
    losses = []
    for images, labels in train_loader:
        # evaluer fonction cout
        labels_predict = model(images)
        loss = F.cross_entropy(labels_predict, labels)
        losses.append(loss.item())

        # mise à jour paramètre
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = np.mean(losses)
    train_losses.append(train_loss)
    
    # validation
    with torch.no_grad():
        for images, labels in valid_loader:
            labels_predict = model(images)
            valid_loss = F.cross_entropy(labels_predict, labels).item()
            valid_losses.append(valid_loss)
    
    # afficher progression
    print('epoch %i train %.2e  valid %.2e' % (epoch, train_loss, valid_loss))


# tracer les courbes de progression train/valid 
plt.semilogy(train_losses)
plt.semilogy(valid_losses)
plt.legend(['train loss', 'valid loss'])
plt.xlabel("epochs")


#%%
# import torch.optim as optim

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# epochs = 10
# for epoch in range(epochs):
#     logging_loss = 0.0
#     for i, data in enumerate(train_loader):
#         input, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         out = model(input)

#         # Compute loss
#         loss = criterion(out, labels)

#         # Compute gradients
#         loss.backward()

#         # Backward pass - model update
#         optimizer.step()

#         logging_loss += loss.item()

#     # Logging training loss
#     logging_loss /= 2000
#     print('Training loss epoch ', epoch, ' -- mini-batch ', i, ': ', logging_loss)
#     logging_loss = 0.0

#     # Model validation
#     with torch.no_grad():
#         logging_loss_val = 0.0
#         for data_val in valid_loader:
#             input_val, labels_val = data_val
#             out_val = model(input_val)
#             loss_val = criterion(out_val, labels_val)
#             logging_loss_val += loss_val.item()
#         logging_loss_val /= len(valid_loader)
#         print('Validation loss: ', logging_loss_val)