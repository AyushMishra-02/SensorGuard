"""
generate_data.py — Generates synthetic NASA CMAPSS-like turbofan engine degradation data.

This creates realistic time-series sensor data mimicking the FD001 dataset structure:
- 100 engine units with varying lifespans (130–350 cycles)
- 3 operational settings + 21 sensor channels
- Gradual degradation trends with Gaussian noise
- Output: data/train_FD001.txt (space-separated, no header)

For real data, download from:
  https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
  or NASA Prognostics Center: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
"""

import numpy as np
import pandas as pd
import os

def generate_cmapss_data(n_units=100, seed=42):
    np.random.seed(seed)
    
    all_data = []
    
    for unit_id in range(1, n_units + 1):
        # Random lifespan between 130 and 350 cycles
        max_cycles = np.random.randint(130, 351)
        cycles = np.arange(1, max_cycles + 1)
        n = len(cycles)
        
        # Operational settings (3 columns) — mostly constant with slight variation
        op1 = np.random.choice([-0.0007, 0.0, 0.0007], size=n) + np.random.normal(0, 0.0001, n)
        op2 = np.random.choice([-0.0004, 0.0, 0.0004], size=n) + np.random.normal(0, 0.0001, n)
        op3 = np.full(n, 100.0) + np.random.normal(0, 0.01, n)
        
        # Degradation factor: increases exponentially near end of life
        degradation = np.zeros(n)
        onset = int(n * 0.6)  # degradation starts at ~60% of life
        remaining = n - onset
        if remaining > 0:
            degradation[onset:] = np.linspace(0, 1, remaining) ** 2
        
        # 21 sensor channels with different characteristics
        sensors = {}
        
        # Sensors with clear degradation trends (useful features)
        trend_sensors = {
            's2':  (641.82, 0.5,  10.0),   # Total temp at LPC outlet
            's3':  (1589.0, 1.0,  15.0),   # Total temp at HPC outlet
            's4':  (1400.0, 2.0,  20.0),   # Total temp at LPT outlet
            's7':  (554.36, 0.2,   5.0),   # Total pressure at HPC outlet
            's8':  (2388.0, 0.5,  -8.0),   # Physical fan speed
            's9':  (9046.0, 2.0, -15.0),   # Physical core speed
            's11': (47.47,  0.1,   2.0),   # Static pressure at HPC outlet
            's12': (521.66, 0.3,   5.0),   # Ratio of fuel flow to Ps30
            's13': (2388.0, 0.5,  -8.0),   # Corrected fan speed
            's14': (8138.0, 1.0, -10.0),   # Corrected core speed
            's15': (8.44,   0.02,  0.5),   # Bypass ratio
            's17': (392.0,  0.3,   3.0),   # Bleed enthalpy
            's20': (38.06,  0.05,  1.0),   # HPT coolant bleed
            's21': (23.42,  0.02,  0.5),   # LPT coolant bleed
        }
        
        # Sensors with little/no degradation signal (constant-ish, noisy)
        flat_sensors = {
            's1':  (518.67, 0.1,  0.0),
            's5':  (14.62,  0.01, 0.0),
            's6':  (21.61,  0.01, 0.0),
            's10': (1.3,    0.001, 0.0),
            's16': (0.03,   0.001, 0.0),
            's18': (2388.0, 0.3,  0.0),
            's19': (100.0,  0.01, 0.0),
        }
        
        for sid, (base, noise_std, trend_mag) in {**trend_sensors, **flat_sensors}.items():
            noise = np.random.normal(0, noise_std, n)
            trend = degradation * trend_mag
            sensors[sid] = base + noise + trend
        
        # Build dataframe for this unit
        unit_data = pd.DataFrame({
            'unit': unit_id,
            'cycle': cycles,
            'op1': op1,
            'op2': op2,
            'op3': op3,
        })
        
        for i in range(1, 22):
            col = f's{i}'
            unit_data[col] = sensors[col]
        
        all_data.append(unit_data)
    
    df = pd.concat(all_data, ignore_index=True)
    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    print("Generating synthetic NASA CMAPSS-like turbofan engine data...")
    df = generate_cmapss_data(n_units=100)
    
    # Save as space-separated file (CMAPSS format)
    filepath = os.path.join('data', 'train_FD001.txt')
    df.to_csv(filepath, sep=' ', header=False, index=False)
    
    print(f"Dataset saved to '{filepath}'")
    print(f"Shape: {df.shape}")
    print(f"Units: {df['unit'].nunique()}")
    print(f"Cycle range: {df['cycle'].min()} – {df['cycle'].max()}")
    print(f"Columns: {list(df.columns)}")
