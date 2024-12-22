import numpy as np

def grad_V(x_tup):
  x = x_tup[0]
  y = x_tup[1]
  x_next = -x + ( np.exp(-1/2*((x-1)**2+y**2)) - np.exp(-1/2*((x+1)**2+y**2)) ) / ( np.exp(-1/2*((x-1)**2+y**2)) + np.exp(-1/2*((x+1)**2+y**2)) )
  y_next = -y
  return np.array([x_next, y_next])

def simulate_exit_time(x_0, grad_V, epsilon, delta, dt):
    x = x_0
    t = 0

    while True:
        dWt = np.random.normal(0, np.sqrt(dt))  # Scaled by sqrt(dt)
        x = x + grad_V(x) * dt + np.sqrt(2 * epsilon) * dWt
        t += dt
        if np.linalg.norm(x) < delta:
            break

    return t

def compute_average_exit_time(x_0, grad_V, epsilon=1, delta=0.1, dt=0.01, iterations=1000):
    tau_list = []

    for _ in range(iterations):
        tau = simulate_exit_time(x_0, grad_V, epsilon, delta, dt)
        tau_list.append(tau)

    average_tau = sum(tau_list) / iterations
    return average_tau


x_0 = (1,0)

print(compute_average_exit_time(x_0, grad_V))


