#import relevant packages 
import numpy as np
import matplotlib.pyplot as plt

def VKF(observations, lambda_ = 0.2, v0 = 1.0, s = 0.3): 
    
    #initialization
    T = len(observations) #num time steps
    m = observations[0]; w = s #posterior mean, posterior variance
    v = v0 #volatility
    
    #values to track 
    predictions = np.full(T,np.nan)
    learning_rate = np.full(T,np.nan)
    volatility = np.full(T,np.nan)
    prediction_error = np.full(T,np.nan)
    volatility_error = np.full(T,np.nan)
    
    #simulate 
    for t in range(T): 
        
        #update stored values
        o = observations[t]
        predictions[t] = m
        volatility[t] = v 
        
        #save prev values
        mpre = m; wpre = w 
        
        #update posterior parameters
        delta_m = o-m #prediction error 
        k = (w+v)/(w+v+s) #Kalman Gain
        m = m + k*delta_m #new prediction
        w = (1-k)*(w+v) #new posterior variance
        
        #update volatility
        wcov = (1-k)*wpre
        delta_v = (m-mpre)**2 + wpre + w - 2*wcov -v
        v = v + lambda_*delta_v
        
        #update remainig tracked values
        learning_rate[t] = k 
        prediction_error[t] = delta_m
        volatility_error[t] = delta_v
        
    #create dictionary with all values
    signals = {
        "predictions": predictions, 
        "volatility": volatility, 
        "learning_rate": learning_rate,
        "prediction_error": prediction_error,
        "volatility_error": volatility_error
    }
    
    return signals