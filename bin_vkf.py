#import relevant packages 
import numpy as np 
import matplotlib.pyplot as plt

def bin_VKF(observations, lambda_ = 0.1, v0 = 0.1, omega = 0.1): 
    
    # Initialization
    T = len(observations)       # Number of time steps
    m = 0                    # Posterior mean (float)
    w0 = omega                  # Initial posterior variance
    w = w0                      # Current posterior variance
    v = v0                      # Volatility
    
    # Values to track
    predictions = np.full(T, np.nan)
    learning_rate = np.full(T, np.nan)
    volatility = np.full(T, np.nan)
    prediction_error = np.full(T, np.nan)
    volatility_error = np.full(T, np.nan)
    
    # Sigmoid function
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    
    # Simulation loop
    for t in range(T):
        # Current observation
        o = observations[t]
        
        # Store current predictions and volatility
        predictions[t] = m
        volatility[t] = v
        
        # Save previous values
        mpre = m
        wpre = w
        
        # Update posterior parameters
        delta_m = o - sigmoid(m)          # Prediction error
        k = (w + v) / (w + v + omega)     # Kalman Gain
        alpha = np.sqrt(w + v)            # Learning rate
        m = m + alpha * delta_m           # Updated mean
        w = (1 - k) * (w + v)             # Updated variance
        
        # Update volatility
        wcov = (1 - k) * wpre
        delta_v = (m - mpre)**2 + w + wpre - 2 * wcov - v
        v = v + lambda_ * delta_v
        
        # Store tracked values
        learning_rate[t] = alpha
        prediction_error[t] = delta_m
        volatility_error[t] = delta_v
    
    # Create signals dictionary
    signals = {
        'predictions': sigmoid(predictions),
        'volatility': volatility,
        'learning_rate': learning_rate,
        'prediction_error': prediction_error,
        'volatility_error': volatility_error
    }
    
    return signals