import numpy as np
from euler_maruyama import *
import matplotlib.pyplot as plt

x_0 = np.array([1,0])

epsilons = np.arange(1.0, 50.0, 1.0)

taus = []
for epsilon in epsilons:
    tau = compute_average_exit_time(x_0, grad_V, epsilon)
    taus.append(tau)
    print(f"Epsilon: {epsilon:.1f}, Tau: {tau}")

# Plot epsilon vs. tau

plt.figure(figsize=(8, 5))
plt.plot(epsilons, taus, marker='o', linestyle='-', color='b')
plt.title("Epsilon vs. Average Exit Time")
plt.xlabel("Epsilon")
plt.ylabel("Average Exit Time (Tau)")
plt.grid(True)
plt.show()
