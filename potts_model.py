import numpy as np
import matplotlib.pyplot as plt

def initialize_lattice(N, q):
    """Initialize the lattice with random spin states."""
    return np.random.randint(0, q, size=(N, N))

def delta_energy(lattice, i, j, q, beta):
    """Calculate the change in energy for flipping a spin."""
    N = lattice.shape[0]
    current_spin = lattice[i, j]
    new_spin = np.random.randint(0, q)
    while new_spin == current_spin:
        new_spin = np.random.randint(0, q)

    neighbors = [
        lattice[(i - 1) % N, j],  # above
        lattice[(i + 1) % N, j],  # below
        lattice[i, (j - 1) % N],  # left
        lattice[i, (j + 1) % N]   # right
    ]

    delta_E = sum(1 if neighbor == new_spin else 0 for neighbor in neighbors) - \
              sum(1 if neighbor == current_spin else 0 for neighbor in neighbors)
    return delta_E * -beta, new_spin

def metropolis_step(lattice, q, beta):
    """Perform one Metropolis step."""
    N = lattice.shape[0]
    for _ in range(N * N):  # Attempt N^2 updates
        i, j = np.random.randint(0, N), np.random.randint(0, N)
        dE, new_spin = delta_energy(lattice, i, j, q, beta)
        if dE <= 0 or np.random.rand() < np.exp(-dE):
            lattice[i, j] = new_spin

def compute_observables(lattice, q):
    """Compute the internal energy and magnetization."""
    N = lattice.shape[0]
    energy = 0
    for i in range(N):
        for j in range(N):
            current_spin = lattice[i, j]
            neighbors = [
                lattice[(i - 1) % N, j],  # above
                lattice[(i + 1) % N, j],  # below
                lattice[i, (j - 1) % N],  # left
                lattice[i, (j + 1) % N]   # right
            ]
            energy -= sum(1 if neighbor == current_spin else 0 for neighbor in neighbors)
    energy /= 2  # Each interaction is counted twice
    energy /= N * N
    return energy

def simulate_potts(N, q, T_range, steps=5000, equil_steps=1000):
    """Simulate the Potts model and compute observables."""
    internal_energies = []
    specific_heats = []
    
    for T in T_range:
        beta = 1 / T
        lattice = initialize_lattice(N, q)
        
        # Equilibration steps
        for _ in range(equil_steps):
            metropolis_step(lattice, q, beta)
        
        # Measurement steps
        energies = []
        for _ in range(steps):
            metropolis_step(lattice, q, beta)
            energy = compute_observables(lattice, q)
            energies.append(energy)
        
        # Compute averages
        avg_energy = np.mean(energies)
        avg_energy_sq = np.mean(np.array(energies) ** 2)
        specific_heat = (avg_energy_sq - avg_energy ** 2) * (N * N) / T ** 2
        
        internal_energies.append(avg_energy)
        specific_heats.append(specific_heat)
    
    return internal_energies, specific_heats

# Parameters
N = 20
q = 3
T_range = np.linspace(0.5, 3.0, 50)

# Simulation
internal_energies, specific_heats = simulate_potts(N, q, T_range)

# Plot results
plt.figure(figsize=(10, 5))

# Internal energy plot
plt.subplot(1, 2, 1)
plt.plot(T_range, internal_energies, marker='o', label="Internal Energy")
plt.xlabel("Temperature (T)")
plt.ylabel("Internal Energy")
plt.title("Internal Energy vs. Temperature")
plt.grid(True)

# Specific heat plot
plt.subplot(1, 2, 2)
plt.plot(T_range, specific_heats, marker='o', label="Specific Heat", color='r')
plt.xlabel("Temperature (T)")
plt.ylabel("Specific Heat")
plt.title("Specific Heat vs. Temperature")
plt.grid(True)

# Save the figure
plt.tight_layout()
plt.savefig("potts_model_plots.png", dpi=300)
plt.close()  # Close the figure after saving to free memory

