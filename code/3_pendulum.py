import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class Pendulum(Dataset):
    g = 10.0
    dt = .05
    max_speed = 4 # speed = d(theta)/dt
    min_speed = -max_speed
    
    def __init__(self, sequences_number = 1, sequences_length = 40):
        self.m = 1.0
        
        # initial conditions
        self.l = np.random.uniform(.5, 3, sequences_number)
        theta = np.random.uniform(np.pi / 2, 3 * np.pi / 2, sequences_number)
        speed = np.random.uniform(self.min_speed, self.max_speed, sequences_number)
        
        # trajectory : we only store theta
        self.thetas = np.zeros((sequences_number, sequences_length), dtype=np.float32)
        self.thetas[:, 0] = theta
        for k in range(1, sequences_length):
            theta, speed = self.step(theta, speed)
            self.thetas[:, k] = theta

    @staticmethod
    def angle_normalize(x):
        return x #x % (2 * np.pi) # ((x + np.pi) % (2 * np.pi)) - np.pi

    def step(self, theta, thdot):
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = 0
        newthdot = thdot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = theta + newthdot * dt
        newth = Pendulum.angle_normalize(newth)

        return newth, newthdot

    def _get_obs(self):
        theta, thetadot = self.theta, self.speed
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    
    def __len__(self):
      return self.thetas.shape[0]

    def __getitem__(self, index):
      return self.l[index], self.thetas[index, :]

pendulum = Pendulum(2, 100)

l, thetas = pendulum[0]
plt.plot(thetas, "g*-")
plt.title('l = %.3f' % l)
plt.show()
l, thetas = pendulum[1]
plt.plot(thetas, "r*-")
plt.title('l = %.3f' % l)
plt.show()
