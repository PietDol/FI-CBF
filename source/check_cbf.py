import matplotlib.pyplot as plt
import numpy as np

def func(x, x_obs=3, w_obs=8):
    y = (x - x_obs)**2 - (w_obs/2)**2
    return y

x = np.linspace(0, 12, 50)
y = func(x)
y1 = 3 * y

plt.figure()
plt.plot(x, y, label='a=1')
plt.plot(x, y1, label='a=3')
plt.grid()
plt.legend()
plt.show()