import numpy as np
import matplotlib.pyplot as plt
from vkf import VKF  # Ensure the VKF module is available

def run_switch_vkf(T=200, s0=10, s_low=0.1, s_high=0.2, v=0.1, l=0.1, v0=0.1, s=0.1, switch_indices=None):
    
    """
    Run a Periodic Switching VKF experiment
    
    
    Parameters: 
    - T (int): Number of time steps
    - s0 (float): Initial State Value
    - s_low (float): Low observations noise std
    - s_high (float): High observation noise std
    - l: Volatilitiy learning rate 
    - v0: Initial Volatility Estimate
    - s: Initial posterior variance estimate 
    - switch_indices (list of int, optional): Time indices for switches in observation noise
    
    Returns: 
    - latent_state (ndarray): True latent state over time
    - observations (ndarray): Observed values with noise
    - signals (dict): Signals output from the VKF
    """ 
    
    #initialize 
    latent_state = np.zeros(T)
    observations = np.zeros(T) 

    #initial latent_state
    latent_state[0] = s0
    
    # Generate latent state trajectory
    for t in range(1, T):
        latent_state[t] = latent_state[t - 1] + np.random.normal(0, v)
        
    # Handle case where no switch indices are provided
    if switch_indices is None:
        switch_indices = [0, T]
        
                
    # Determine the volatility phase for each time step
    high_volatility = True  # Start with high volatility
    current_switch_index = 0  # Track position in switch_indices
    
    for i in range(T):
        # Check if we've reached the next switch point
        if current_switch_index < len(switch_indices) and i >= switch_indices[current_switch_index]:
            high_volatility = not high_volatility  # Toggle volatility
            current_switch_index += 1
        
        # Choose noise based on current volatility phase
        if high_volatility:
            noise = np.random.normal(0, s_high)
        else:
            noise = np.random.normal(0, s_low)
        
        # Generate observation with selected noise
        observations[i] = latent_state[i] + noise
    
    # Run VKF
    signals = VKF(observations, l, v0, s)
    
    return latent_state, observations, signals
    