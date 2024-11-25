import numpy as np
from scipy.stats import pearsonr, spearmanr
from run_switch_vkf import run_switch_vkf  # Ensure run_vkf is in the same directory or installed as a module

def lambda_rolling_average_analysis(
    T=200, s0=10, s_low=0.1, s_high=0.2, v=0.1,
    v0=0.1, s=0.1, switch_indices=None, l_num=200
):
    """
    Analyze the relationship between rolling averages and volatility for different lambda values.

    Parameters:
    - T (int): Number of time steps
    - s0 (float): Initial state value
    - s_low (float): Low volatility noise standard deviation
    - s_high (float): High volatility noise standard deviation
    - v (float): State transition noise standard deviation
    - v0 (float): Initial variance of the VKF
    - s (float): Sensory noise in the VKF
    - switch_indices (list of int, optional): Time indices for volatility switches
    - l_num (int): Number of lambda values to test

    Returns:
    - l_vals (ndarray): Array of lambda values
    - correlations (dict): Dictionary containing Pearson and Spearman correlation results
    """
    # Log-spaced lambda values to test 
    l_vals = np.logspace(np.log10(1e-4), np.log10(1.0), l_num)
    
    # Values to store 
    pear_ind = np.full(l_num, np.nan)
    spear_ind = np.full(l_num, np.nan)
    pear_vals = np.full(l_num, np.nan)
    spear_vals = np.full(l_num, np.nan)
    
    for i, l in enumerate(l_vals): 
        # Run VKF
        _, _, signals = run_switch_vkf(
            T=T, s0=s0, s_low=s_low, s_high=s_high, v=v,
            l=l, v0=v0, s=s, switch_indices=switch_indices
        )
        
        # Get relevant values 
        vol = signals['volatility']  # Volatility estimates
        abs_delta = abs(signals['prediction_error'])  # Absolute value of prediction error
        
        # Initialize Pearson and Spearman correlations 
        pearson_corr = []
        spearman_corr = [] 
        
        # Find optimal values 
        for window_size in range(1, int(T / 3)): 
            # Calculate rolling average
            ra = np.convolve(abs_delta, np.ones(window_size) / window_size, mode='valid')
            
            # Store correlation values 
            pearson_corr.append(pearsonr(vol[window_size-1:], ra)[0])
            spearman_corr.append(spearmanr(vol[window_size-1:], ra)[0])
        
        # Update values 
        pear_ind[i] = np.where(pearson_corr == max(pearson_corr))[0][0]
        spear_ind[i] = np.where(spearman_corr == max(spearman_corr))[0][0]
        pear_vals[i] = pearson_corr[int(pear_ind[i])]
        spear_vals[i] = spearman_corr[int(spear_ind[i])]
        
    correlations = {
        "pearson_indexes": pear_ind,
        "pearson_values": pear_vals,
        "spearman_indexes": spear_ind,
        "spearman_values": spear_vals
    }
        
    return l_vals, correlations
