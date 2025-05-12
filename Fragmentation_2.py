import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

def run_1D_simulation_time(E, Gc, strength_base, density, strain_rate, 
                           N=20000, L=0.5, max_steps=100000, tolerance=2000, save_interval=1000):
    """
    Runs the 1D cohesive fracture simulation and records snapshots of the state variables over time.
    
    Parameters:
      E            : Young's modulus (Pa)
      Gc           : Fracture energy (J/m^2)
      strength_base: Base material strength (Pa)
      density      : Material density (kg/m^3)
      strain_rate  : Applied strain rate (1/s)
      N            : Number of spatial nodes
      L            : Length of the bar (m)
      max_steps    : Maximum number of time steps
      tolerance    : Convergence tolerance (number of consecutive steps with no new damage)
      save_interval: Save snapshot every this many steps
      
    Returns:
      A dictionary with:
         'X'            : Spatial coordinate array (length N)
         'time'         : Array of saved time points
         'stress'       : 2D array of stress snapshots (shape: [num_snapshots, N])
         'velocity'     : 2D array of net velocity snapshots (shape: [num_snapshots, N])
         'damage'       : 2D array of damage snapshots (shape: [num_snapshots, N])
         'COD_MAX'      : 2D array of COD_MAX snapshots (shape: [num_snapshots, N])
         'frag_sizes'   : Final fragment sizes (in meters)
         'num_frag'     : Final number of fragments (number of nodes with Damage >= 1)
         'avg_frag_size': Final average fragment size (in micrometers)
    """
    # Spatial discretization and derived quantities
    X = np.linspace(0, L, N)
    dx = X[1] - X[0]
    rho = density
    c = np.sqrt(E / rho)
    Delta_Time = dx / c

    # Material properties and initial fields
    Strength = strength_base * np.ones(N)
    COD_FAIL = 2 * Gc / Strength

    stress = 0.99 * strength_base * np.ones(N)
    Damage = np.zeros(N)
    COD = np.zeros(N)
    COD_MAX = np.zeros(N)

    # Velocity initialization (far-field loading)
    velocity_left_end = -strain_rate * L
    velocity_right_end = strain_rate * L
    velocity_pos = np.linspace(velocity_left_end, velocity_right_end, N)
    velocity_neg = np.copy(velocity_pos)

    # Precompute interior indices and constants
    indices = np.arange(1, N-1)
    inv_rho_c = 1 / (rho * c)
    inv_2rho_c = 1 / (2 * rho * c)

    # Lists to store snapshots
    time_list = []
    stress_list = []
    velocity_list = []
    damage_list = []
    COD_MAX_list = []

    step = 1
    prev_break_count = np.count_nonzero(Damage >= 1)
    no_break_counter = 0

    while step < max_steps:
        stressNEW = stress.copy()
        velocityNEW_pos = velocity_pos.copy()
        velocityNEW_neg = velocity_neg.copy()

        # -------------------------------
        # Boundary Conditions
        # -------------------------------
        stressNEW[0] = min(Strength[0] * (1 - Damage[0]),
                           stress[1] - rho * c * (velocity_left_end - velocity_neg[1]))
        velocityNEW_pos[0] = velocity_left_end
        velocityNEW_neg[0] = velocity_left_end

        stressNEW[-1] = stress[-2] + rho * c * (velocity_right_end - velocity_pos[-2])
        velocityNEW_pos[-1] = velocity_right_end
        velocityNEW_neg[-1] = velocity_right_end

        # -------------------------------
        # Interior Nodes Update (Vectorized)
        # -------------------------------
        s_left = stress[indices - 1]
        s_right = stress[indices + 1]
        vpos_left = velocity_pos[indices - 1]
        vneg_right = velocity_neg[indices + 1]

        stress_interior = 0.5 * (s_left + s_right) + 0.5 * rho * c * (vneg_right - vpos_left)
        allowed = Strength[indices] * (1 - Damage[indices])
        
        v_new_pos = np.empty_like(stress_interior)
        v_new_neg = np.empty_like(stress_interior)
        
        mask = stress_interior < allowed
        
        v_new = 0.5 * (vpos_left + vneg_right) + inv_2rho_c * (s_right - s_left)
        v_new_pos[mask] = v_new[mask]
        v_new_neg[mask] = v_new[mask]
        
        stress_interior[~mask] = allowed[~mask]
        v_new_pos[~mask] = inv_rho_c * (s_right[~mask] - allowed[~mask]) + vneg_right[~mask]
        v_new_neg[~mask] = inv_rho_c * (allowed[~mask] - s_left[~mask]) + vpos_left[~mask]
        
        if np.any(~mask):
            delta_COD = (Delta_Time / 2) * (
                v_new_pos[~mask] + velocity_pos[indices[~mask]] -
                v_new_neg[~mask] - velocity_neg[indices[~mask]]
            )
            delta_COD = np.maximum(0, delta_COD)
            COD[indices[~mask]] += delta_COD
            COD_MAX[indices[~mask]] = np.maximum(COD[indices[~mask]], COD_MAX[indices[~mask]])
            Damage[indices[~mask]] = np.clip(COD_MAX[indices[~mask]] / COD_FAIL[indices[~mask]], 0, 1)

        stressNEW[indices] = stress_interior
        velocityNEW_pos[indices] = v_new_pos
        velocityNEW_neg[indices] = v_new_neg
        
        stress = stressNEW
        velocity_pos = velocityNEW_pos
        velocity_neg = velocityNEW_neg

        # Save snapshot every save_interval steps.
        if step % save_interval == 0:
            current_time = step * Delta_Time
            time_list.append(current_time)
            stress_list.append(stress.copy())
            net_velocity = 0.5 * (velocity_pos + velocity_neg)
            velocity_list.append(net_velocity.copy())
            damage_list.append(Damage.copy())
            COD_MAX_list.append(COD_MAX.copy())

        new_break_count = np.count_nonzero(Damage >= 1)
        if new_break_count > 0:
            if new_break_count == prev_break_count:
                no_break_counter += 1
            else:
                no_break_counter = 0
            prev_break_count = new_break_count
            if no_break_counter >= tolerance:
                print(f"Converged after {step} steps.")
                break

        step += 1

    # Ensure final state is saved.
    final_time = step * Delta_Time
    if len(time_list) == 0 or time_list[-1] < final_time:
        time_list.append(final_time)
        stress_list.append(stress.copy())
        net_velocity_final = 0.5 * (velocity_pos + velocity_neg)
        velocity_list.append(net_velocity_final.copy())
        damage_list.append(Damage.copy())
        COD_MAX_list.append(COD_MAX.copy())

    # -------------------------------
    # Post-Processing: Fragment Size Distribution
    # -------------------------------
    break_indices = np.where(Damage >= 1)[0]
    if break_indices.size == 0:
        frag_sizes = np.array([L])
    else:
        indices_frag = np.concatenate(([0], break_indices, [N - 1]))
        frag_sizes = np.diff(indices_frag) * dx

    num_frag = break_indices.size
    avg_frag_size = (L / num_frag) * 1e6 if num_frag > 0 else L  # in micrometers

    net_velocity_final = 0.5 * (velocity_pos + velocity_neg)

    return {
        'X': X,
        'time': np.array(time_list),
        'stress': np.array(stress_list),
        'velocity': np.array(velocity_list),
        'damage': np.array(damage_list),
        'COD_MAX': np.array(COD_MAX_list),
        'frag_sizes': frag_sizes,
        'num_frag': num_frag,
        'avg_frag_size': avg_frag_size
    }
