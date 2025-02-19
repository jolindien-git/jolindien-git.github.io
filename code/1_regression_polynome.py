import numpy as np
from matplotlib import pyplot as plt


#%% Générer les données
N = 30 # nombre de points
x = np.random.uniform(0, 1, size=N)
y = np.sin(2 * np.pi * x) + np.random.normal(scale=.1, size=N)

# visualiser données
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('N=%i' % N)
plt.show()


#%% convertir données (numpy -> pytorch)
import torch
x, y = torch.tensor(x), torch.tensor(y)

# initialise theta
M = 3
theta = torch.zeros(M + 1, requires_grad = True)

# fonction polynome
def get_predict(x, coefs):
  y_predict = torch.zeros_like(x)
  for i in range(len(theta)):
    y_predict = y_predict + coefs[i] * x**i
  return y_predict

# fonction coût
def get_loss(y_predict):
  return ((y_predict - y)**2).mean()


#%% descente de gradient
lr = .5 # learning rate
EPOCHS = M * 10000 # nombre d'iterations
for k in range(EPOCHS + 1):
  loss = get_loss(get_predict(x, theta))
  if k % 1000 == 0:
    print("k=%i \t L(x, theta)=%.3e \t lr=%.3e" % (k, loss.item(), lr))
  grad = torch.autograd.grad(loss, theta)[0]
  theta = theta - lr * grad # mise à jour de theta
  if k == EPOCHS - EPOCHS // 10:
      lr /= 10

# visualiser résultat
def print_polynome(coefs):
    poly_str = ' + '.join(["%.3f x^%i" % (v, i) for i, v in enumerate(coefs)])
    print("polynome:", poly_str)

print_polynome(theta)

plt.plot(x, y, 'o') # données
x_predict = np.linspace(0, 1, num=200)
y_predict = get_predict(torch.tensor(x_predict), theta.detach())
plt.plot(x_predict, y_predict) # predictions
plt.title('M = %i' % M)
plt.legend(['données', 'polynome'])