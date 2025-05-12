import numpy as np
import matplotlib.pyplot as plt
from Fragmentation_2 import run_1D_simulation_time
import pickle




materials = {
    "Aluminum Alloy": {
        "E": (60e9, 75e9),         # Pa
        # Ductile aluminum alloys can have effective fracture energies on the order of 10^3 J/m².
        "Gc": (5000, 25000),         # J/m²
        "strength": (200e6, 350e6),  # Pa
        "density": (2700, 2800)      # kg/m³
    },
    "Titanium Alloy": {
        "E": (100e9, 120e9),        # Pa
        # Titanium alloys (also ductile) typically have slightly higher values.
        "Gc": (20000, 90000),         # J/m²
        "strength": (800e6, 1100e6),  # Pa
        "density": (4430, 4500)      # kg/m³
    },
    "CFRP": {
        "E": (50e9, 120e9),         # Pa
        # For composite laminates, the Mode I (interlaminar) fracture energy is often in the
        # few-hundred J/m² range.
        "Gc": (100, 900),           # J/m²
        "strength": (300e6, 800e6),   # Pa
        "density": (1600, 2000)      # kg/m³
    },
    "Stainless Steel": {
        "E": (190e9, 210e9),        # Pa
        # Ductile stainless steels can require ~10^3–10^4 J/m² to propagate a crack.
        "Gc": (12000, 110000),         # J/m²
        "strength": (500e6, 1000e6),  # Pa
        "density": (7800, 8000)      # kg/m³
    },
    "Beryllium": {
        "E": (287e9, 310e9),        # Pa
        # Beryllium is brittle so its effective fracture energy remains low,
        # not far above the intrinsic (tens of J/m²).
        "Gc": (30, 350),             # J/m²
        "strength": (240e6, 400e6),   # Pa
        "density": (1850, 1900)      # kg/m³
    },
    "Glass": {
        "E": (60e9, 80e9),          # Pa
        # Glass is also brittle; based on typical fracture toughness values,
        # the effective Gc is on the order of several J/m².
        "Gc": (7, 14),              # J/m²
        "strength": (40e6, 100e6),    # Pa
        "density": (2400, 2600)      # kg/m³
    },
    "Copper Alloy": {
        "E": (110e9, 130e9),        # Pa
        # Ductile copper alloys are similar to aluminum in that the plastic work
        # during fracture boosts the effective fracture energy.
        "Gc": (8000, 60000),         # J/m²
        "strength": (300e6, 600e6),   # Pa
        "density": (8800, 9000)      # kg/m³
    }
}


def main():
    # -------------------------------
    # Global Sensitivity Analysis Setup
    # -------------------------------
    N_sim = 5          # Number of simulations per material sample
    applied_bar_strain = 1
    L = 0.05           # Bar length in meters
    N = 50000          # Number of spatial nodes
    
    strain_rates = np.linspace(1e6, 1e7, 20)  # Strain rates (1/s)
    
    # global_results will be organized by strain_rate then by material.
    global_results = {}
    
    # Assume 'materials' is defined externally, for example:
    # materials = {
    #    "MaterialA": {"E": [min_E, max_E], "Gc": [min_Gc, max_Gc],
    #                  "strength": [min_strength, max_strength], "density": [min_density, max_density]},
    #    "MaterialB": {...}
    # }
    
    for strain_rate in strain_rates:
        global_results[strain_rate] = {}
        for material, props in materials.items():
            # Generate samples uniformly for each parameter within its range.
            E_samples = np.random.uniform(props["E"][0], props["E"][1], N_sim)
            Gc_samples = np.random.uniform(props["Gc"][0], props["Gc"][1], N_sim)
            strength_samples = np.random.uniform(props["strength"][0], props["strength"][1], N_sim)
            density_samples = np.random.uniform(props["density"][0], props["density"][1], N_sim)
            
            # Pre-allocate arrays for aggregated (scalar) outputs.
            out_num_frag = np.zeros(N_sim)
            out_avg_frag = np.zeros(N_sim)
            
            # Lists to store the full field outputs for each simulation.
            out_time = []      # List of time arrays (one per simulation)
            out_stress = []    # List of stress arrays (each is a 2D array: [num_snapshots, N])
            out_velocity = []  # List of net velocity arrays (each is 2D: [num_snapshots, N])
            out_damage = []    # List of damage arrays (each is 2D: [num_snapshots, N])
            out_COD_MAX = []   # List of COD_MAX arrays (each is 2D: [num_snapshots, N])
            out_frag_sizes = []# List of fragment size distributions (each array)
            
            # Loop over the samples.
            for i in range(N_sim):
                E_val = E_samples[i]
                Gc_val = Gc_samples[i]
                strength_val = strength_samples[i]
                density_val = density_samples[i]
                
                # Call the simulation that returns the full time evolution as a dictionary.
                sim_out = run_1D_simulation_time(
                    E=E_val,
                    Gc=Gc_val,
                    strength_base=strength_val,
                    density=density_val,
                    strain_rate=strain_rate,
                    N=N,
                    L=L,
                    tolerance=5000,
                    save_interval=1000  # Adjust as needed
                )
                
                # Append full arrays.
                out_time.append(sim_out['time'])
                out_stress.append(sim_out['stress'])
                out_velocity.append(sim_out['velocity'])
                out_damage.append(sim_out['damage'])
                out_COD_MAX.append(sim_out['COD_MAX'])
                
                # Save aggregated outputs.
                out_frag_sizes.append(sim_out['frag_sizes'])
                out_num_frag[i] = sim_out['num_frag']
                out_avg_frag[i] = sim_out['avg_frag_size']
            
            # Save results for this material at the current strain rate.
            global_results[strain_rate][material] = {
                "E_samples": E_samples,
                "Gc_samples": Gc_samples,
                "strength_samples": strength_samples,
                "density_samples": density_samples,
                "num_frag": out_num_frag,
                "avg_frag": out_avg_frag,
                "time": out_time,              # Full time arrays.
                "stress": out_stress,          # Full stress arrays.
                "velocity": out_velocity,      # Full net velocity arrays.
                "damage": out_damage,          # Full damage arrays.
                "COD_MAX": out_COD_MAX,        # Full COD_MAX arrays.
                "frag_sizes": out_frag_sizes,  # Fragment size distributions.
                "strain_rate": strain_rate,
            }
            print(f"{material} done at strain_rate = {strain_rate:.2e}: mean avg_frag = {np.mean(out_avg_frag):.2e} m, mean num_frag = {np.mean(out_num_frag):.2f}")
        
        # Save results for this strain_rate.
        with open(f"global_sensitivity_results_{strain_rate:.2e}_0407.pkl", "wb") as f:
            pickle.dump(global_results[strain_rate], f)
        print(f"Global sensitivity results for strain_rate = {strain_rate:.2e} have been saved.")

if __name__ == "__main__":
    main()