import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 40  # Lattice size N x N
q = 3   # Number of states in Potts model
T_range = [2.0]  # Temperatures
#h_range = np.linspace(-1.0, 1.0, 100)  # Range of h values
h_range = [0]
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

# Run Monte Carlo simulation for a specific temperature and h
def run_simulation(N, T, h, equil_steps, mc_steps):
    lattice = initialize_lattice(N)
    beta = 1 / T
    magnetization = compute_magnetization(lattice)
    
    # Store samples for correlation function calculation
    samples = []
    
    # Equilibrate the system
    for _ in range(equil_steps):
        lattice, magnetization = metropolis_update(lattice, beta, h, magnetization)
    
    # Collect data for magnetization and correlation function
    for _ in range(mc_steps):
        lattice, magnetization = metropolis_update(lattice, beta, h, magnetization)
        samples.append(lattice.copy())
    
    return np.mean(magnetization), samples

# Compute correlation function C(i,j)
def compute_correlation(samples, N):
    # Flatten lattice into 1D for each sample
    flat_samples = np.array([sample.flatten() for sample in samples])
    
    # Compute <σ_i> for each site
    avg_spins = np.mean(flat_samples, axis=0)
    
    # Calculate correlation matrix C(i,j)
    C = np.dot(flat_samples.T, flat_samples) / len(flat_samples) - np.outer(avg_spins, avg_spins)
    
    return C

# Compute the spatial correlation function for distance k
def compute_spatial_correlation(C, N, k):
    correlation_function = 0
    
    # Iterate over all sites (i)
    for i in range(N**2):
        # Calculate horizontal and vertical neighbors at distance k
        # Horizontal neighbors
        ni = (i + k) % (N**2)  # Right (periodic boundary)
        nj = (i - k) % (N**2)  # Left (periodic boundary)
        
        # Vertical neighbors
        ni_vert = (i + k * N) % (N**2)  # Down (periodic boundary)
        nj_vert = (i - k * N) % (N**2)  # Up (periodic boundary)
        
        # Add the correlation for these neighbors (horizontal and vertical)
        correlation_function += C[i, ni] + C[i, nj] + C[i, ni_vert] + C[i, nj_vert]
    
    # Normalize the correlation function by the number of pairs
    correlation_function /= (4*N**2)
    
    return correlation_function

# Main loop: Compute magnetization and spatial correlation for different values of h and T
magnetizations_dict = {T: [] for T in T_range}
correlation_lengths = []

# Run for T = 1.5, 2.0, 2.5 for magnetization and spatial correlation function
for T in T_range:
    # Magnetization vs h
    magnetizations_for_h = []
    for h in h_range:
        mag, samples = run_simulation(N, T, h, equil_steps, mc_steps)
        print(f"Running simulation for T = {T}, h = {h}") 
        magnetizations_for_h.append(mag)
        
        # Compute correlation function for h = 0
        if h == 0:
            C = compute_correlation(samples, N)
        
            # Calculate spatial correlation for different k values
            correlation_function = np.zeros(N//2)
            for k in range(1, N//2):
                correlation_function[k] = compute_spatial_correlation(C, N, k)
        
            print(f"Spatial correlation function for T = {T}, h = {h}: {correlation_function}")
            
            # Plot and save spatial correlation function
            plt.plot(range(1, N//2), correlation_function[1:], marker='o', linestyle='-', color='r')
            plt.title(f"Spatial Correlation Function (T={T}, h={h})")
            plt.xlabel("Distance k")
            plt.ylabel("Γ(k)")
            plt.grid(True)
            plt.savefig(f"correlation_function_T_{T}_h_{h}.png")
            plt.clf()  # Clear the figure after saving
    
    magnetizations_dict[T] = magnetizations_for_h

# Plot magnetization vs h for each temperature and save each figure
for T in T_range:
    plt.plot(h_range, magnetizations_dict[T], label=f'T = {T}')
    # Save each plot with a filename including the temperature
    plt.title(f"Magnetization vs. External Field (h) for T = {T}")
    plt.xlabel("External Field (h)")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f"magnetization_vs_h_T_{T}.png")
    plt.clf()  # Clear the figure after saving
