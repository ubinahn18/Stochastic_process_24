import numpy as np
from euler_maruyama import *
import matplotlib.pyplot as plt

initial_x_coords = np.arange(-1.6, 0.2, 1.6)

epsilon = 1.0
taus = []
for x_coord in initial_x_coords:
    tau = compute_average_exit_time(np.array([x_coord,0]), grad_V, epsilon)
    taus.append(tau)
    print(f"Initial x coordinate: {x_coord:.1f}, Tau: {tau}")

# Plot epsilon vs. tau

plt.figure(figsize=(8, 5))
plt.plot(initial_x_coords, taus, marker='o', linestyle='-', color='b')
plt.title("Initial x vs. Average Exit Time")
plt.xlabel("Initial x")
plt.ylabel("Average Exit Time (Tau)")
plt.grid(True)
plt.savefig("initial_x_vs_exit_time.png", dpi=300, bbox_inches="tight")
plt.show()


initial_y_coords = np.arange(-1.0, 0.2, 1.0)

taus = []
for y_coord in initial_y_coords:
    tau = compute_average_exit_time(np.array([0,y_coord]), grad_V, epsilon)
    taus.append(tau)
    print(f"Initial y coordinate: {y_coord:.1f}, Tau: {tau}")

# Plot epsilon vs. tau

plt.figure(figsize=(8, 5))
plt.plot(initial_y_coords, taus, marker='o', linestyle='-', color='b')
plt.title("Initial y vs. Average Exit Time")
plt.xlabel("Initial y")
plt.ylabel("Average Exit Time (Tau)")
plt.grid(True)
plt.savefig("initial_y_vs_exit_time.png", dpi=300, bbox_inches="tight")
plt.show()
