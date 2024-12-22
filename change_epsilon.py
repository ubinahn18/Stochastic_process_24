import numpy as np
from euler_maruyama import *
import matplotlib.pyplot as plt

x_0 = np.array([1,0])

epsilons = np.arange(1.0, 4.0, 0.1)

taus = [compute_average_exit_time(x_0, grad_V, epsilon) for epsilon in epsilons]

# Plot epsilon vs. tau

plt.figure(figsize=(8, 5))
plt.plot(epsilons, taus, marker='o', linestyle='-', color='b')
plt.title("Epsilon vs. Average Exit Time")
plt.xlabel("Epsilon")
plt.ylabel("Average Exit Time (Tau)")
plt.grid(True)
plt.show()
