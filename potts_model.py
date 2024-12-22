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

def metropolis_step_optimized(lattice, beta, energy):
    """Perform a single Metropolis step on the lattice with optimized energy calculation."""
    N = lattice.shape[0]
    x, y = np.random.randint(0, N), np.random.randint(0, N)
    
    current_spin = lattice[x, y]
    new_spin = np.random.randint(1, q + 1)
    while new_spin == current_spin:
        new_spin = np.random.randint(1, q + 1)
    
    # Calculate local energy change
    neighbors = [
        lattice[(x - 1) % N, y],  # Up
        lattice[(x + 1) % N, y],  # Down
        lattice[x, (y - 1) % N],  # Left
        lattice[x, (y + 1) % N]   # Right
    ]
    
    delta_energy = 0
    for neighbor in neighbors:
        delta_energy += int(new_spin != neighbor) - int(current_spin != neighbor)

    
    # Metropolis acceptance criterion
    if delta_energy <= 0 or np.random.rand() < np.exp(-beta * delta_energy):
        lattice[x, y] = new_spin
        energy += delta_energy
    
    return lattice, energy


def simulate_potts(N, q, T_range, equil_steps = 5000, steps = 5000):
    """Simulate the 2D Potts model with optimized energy calculations."""
    internal_energies = []
    specific_heats = []
    
    for T in T_range:
        print(f"Sampling with temperature T = {T}")
        beta = 1 / T
        lattice = np.random.randint(1, q + 1, size=(N, N))
        
        # Initialize total energy
        energy = 0
        for x in range(N):
            for y in range(N):
                spin = lattice[x, y]
                neighbors = [
                    lattice[(x - 1) % N, y],  # Up
                    lattice[(x + 1) % N, y],  # Down
                    lattice[x, (y - 1) % N],  # Left
                    lattice[x, (y + 1) % N]   # Right
                ]
                energy += sum(spin != neighbor for neighbor in neighbors)
        energy /= 2  # Each pair is counted twice
        
        # Equilibration
        for _ in range(equil_steps):
            lattice, energy = metropolis_step_optimized(lattice, beta, energy)
        
        # Measurement
        energies = []
        for _ in range(steps):
            lattice, energy = metropolis_step_optimized(lattice, beta, energy)
            energies.append(energy)
        
        # Compute observables
        avg_energy = np.mean(energies) / (N * N)
        internal_energies.append(avg_energy)
        
        heat_capacity = beta ** 2 * np.var(energies) / (N * N)
        specific_heats.append(heat_capacity)
    
    return internal_energies, specific_heats


# Parameters
N = 30
q = 3
T_range = np.linspace(0.1, 2.5, 120)

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
plt.show()  

