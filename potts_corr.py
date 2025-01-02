import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Parameters
N = 16  # Lattice size N x N
q = 10   # Number of states in Potts model

equil_steps = 5000  # Number of equilibration steps
mc_steps = 5000   # Number of Monte Carlo steps

# Initialize lattice (random spin configuration)
def initialize_lattice(N):
    return np.random.randint(0, q, (N, N))

# Compute magnetization
def compute_magnetization(lattice):
    return np.sum(lattice) / (lattice.shape[0] * lattice.shape[1])

# Metropolis-Hastings update with efficient tracking
def metropolis_update(lattice, beta, h, magnetization):
    N = lattice.shape[0]
    i, j = np.random.randint(0, N, 2)
    current_spin = lattice[i, j]
    
    # Randomly pick a new spin value
    new_spin = np.random.randint(0, q)
    
    # Calculate energy change due to the update
    delta_energy = 0
    neighbors = [
        lattice[(i+1) % N, j], lattice[(i-1) % N, j], 
        lattice[i, (j+1) % N], lattice[i, (j-1) % N]
    ]
    
    # Interaction term (delta(s_i, s_j)) is 1 if spins are the same, 0 otherwise
    for neighbor in neighbors:
        delta_energy += int((new_spin != neighbor)) - int((current_spin != neighbor))
    
    # External field term
    delta_energy += (current_spin - new_spin) * h
    
    # Metropolis decision
    if delta_energy < 0 or np.random.rand() < np.exp(-beta * delta_energy):
        lattice[i, j] = new_spin
        
        # Update magnetization
        magnetization += (new_spin - current_spin) / (N * N)
    
    return lattice, magnetization


def run_simulation(N, T, h, equil_steps, mc_steps):
    lattice = initialize_lattice(N)
    beta = 1 / T
    magnetization = compute_magnetization(lattice)
    
    samples = []
    
    for _ in range(equil_steps):
        lattice, magnetization = metropolis_update(lattice, beta, h, magnetization)
    
    for _ in range(mc_steps):
        lattice, magnetization = metropolis_update(lattice, beta, h, magnetization)
        samples.append(lattice.copy())
    
    return np.mean(magnetization), samples


def compute_correlation(samples, N):
    # Flatten lattice into 1D for each sample
    flat_samples = np.array([sample.flatten() for sample in samples])
    
    # Compute <σ_i> for each site
    avg_spins = np.mean(flat_samples, axis=0)
    
    # Calculate correlation matrix C(i,j)
    C = np.dot(flat_samples.T, flat_samples) / len(flat_samples) - np.outer(avg_spins, avg_spins)
    
    return C


def compute_spatial_correlation(C, N, k):
    correlation_function = 0
    
    # Iterate over all sites (i)
    for i in range(N**2):
        
        # Horizontal neighbors
        ni = (i + k) % (N**2)  # Right (periodic boundary)
        nj = (i - k) % (N**2)  # Left (periodic boundary)
        
        # Vertical neighbors
        ni_vert = (i + k * N) % (N**2)  # Down (periodic boundary)
        nj_vert = (i - k * N) % (N**2)  # Up (periodic boundary)
        
        correlation_function += C[i, ni] + C[i, nj] + C[i, ni_vert] + C[i, nj_vert]

    correlation_function /= (4*N**2)
    
    return correlation_function

# fit correlation function
def exponential_function(k, Gamma_0, xi):
    return Gamma_0 * np.exp(-k / xi)

T_range = [0.5, 1.0, 1.5, 2.0, 2.5]
h_range = np.linspace(0.0, 2.0, 100)

output_directory = r'Stochastic_process_results'

# Ensure the output directory exists
import os
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

magnetizations_dict = {T: [] for T in T_range}

# Run simulation
for T in T_range:
    print(f"Running simulation for T = {T}, Magnetization") 
    magnetizations_for_h = []
    for h in h_range:
        mag, samples = run_simulation(N, T, h, equil_steps, mc_steps)
        magnetizations_for_h.append(mag)
    magnetizations_dict[T] = magnetizations_for_h

# Plot Magnetization
plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size

for T in T_range:
    plt.plot(h_range, magnetizations_dict[T], label=f'T = {T}')

plt.title("Magnetization vs. External Field (h) for Various Temperatures")
plt.xlabel("External Field (h)")
plt.ylabel("Magnetization")
plt.legend(title="Temperatures")  # Add a title to the legend for clarity
plt.grid(True)
plt.savefig(f"{output_directory}/magnetization_vs_h_all_T.png")
plt.clf()


# Correlation function
T_range = np.linspace(0.1, 2.0, num=20)
h = 0
xi_fits = []

for T in T_range:
    mag, samples = run_simulation(N, T, h, equil_steps, mc_steps)
    print(f"Running simulation for T = {T}, Correlation") 
    C = compute_correlation(samples, N)
    correlation_function = np.zeros(N//2)
    for k in range(1, N//2):
        correlation_function[k] = compute_spatial_correlation(C, N, k)
    
    # Plot correlation function for the first T
    if np.isclose(T, T_range[0]):
        plt.plot(range(1, N//2), correlation_function[1:], marker='o', linestyle='-', color='r')
        plt.title(f"Spatial Correlation Function (T={T}, h={h})")
        plt.xlabel("Distance k")
        plt.ylabel("Γ(k)")
        plt.grid(True)
        plt.savefig(f"{output_directory}/correlation_function_T_{T}_h_{h}.png")
        plt.clf()  

    # Fit the exponential function
    k_values = np.arange(1, N//2)
    log_correlation = np.log(correlation_function[1:])

    slope, intercept = np.polyfit(k_values, log_correlation, 1)
    xi_fit = -1 / slope  # Since slope = -1/xi
    xi_fits.append(xi_fit)


# Plot T vs xi
plt.plot(T_range, xi_fits, marker='o', linestyle='-', label='ξ')
plt.title("Correlation Length (ξ) vs. Temperature (T)")
plt.xlabel("Temperature (T)")
plt.ylabel("Correlation Length (ξ)")
plt.grid(True)
plt.savefig(f"{output_directory}/T_vs_xi.png")
plt.clf()
