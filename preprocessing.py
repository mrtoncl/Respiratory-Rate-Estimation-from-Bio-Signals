import numpy as np
from scipy.signal import butter, filtfilt

def remove_baseline_wander(signal, fs, cutoff=0.05, order=3):
    """
    Removes low-frequency noise (breathing, motion artifacts) using a High-pass filter.
    UPDATED: Increased default order to 3 for sharper cutoff characteristics.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    
    # Stability check for filter parameters
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    
    # Changed btype to 'high' for removing baseline wander
    b, a = butter(order, normalized_cutoff, btype='high')
    return filtfilt(b, a, signal)  # Zero-phase filtering to prevent signal distortion

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies Butterworth Bandpass filter to isolate the relevant frequency band.
    Standard for physiological signals to remove powerline noise and high-freq artifacts.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Safety checks for digital filter boundaries
    if low <= 0: low = 0.001
    if high >= 1: high = 0.999
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalize_signal(signal, method='minmax'):
    """
    Normalizes the signal amplitude to a specific range for ML model stability.
    Supports Min-Max scaling (0-1) and Z-Score standardization.
    """
    signal = np.array(signal)
    epsilon = 1e-6  # Added small epsilon to prevent ZeroDivisionError
    
    if method == 'minmax':
        sig_min = np.min(signal)
        sig_max = np.max(signal)
        
        # Robust division with epsilon
        if (sig_max - sig_min) < epsilon:
            return np.zeros_like(signal)
        return (signal - sig_min) / (sig_max - sig_min + epsilon)
        
    elif method == 'zscore':
        sig_mean = np.mean(signal)
        sig_std = np.std(signal)
        
        # Robust division with epsilon
        if sig_std < epsilon:
            return np.zeros_like(signal)
        return (signal - sig_mean) / (sig_std + epsilon)
    
    return signal

def preprocess_signal_pipeline(signal, fs):
    """
    Main pipeline function:
    1. Detrending (Baseline Wander Removal)
    2. Bandpass Filtering (0.05 - 0.6 Hz)
    3. Normalization (MinMax with epsilon stability)
    """
    print("  [INFO] Starting filtering process...")
    
    # Step 1: Remove Baseline Wander (Optimized Order)
    sig_no_wander = remove_baseline_wander(signal, fs, cutoff=0.05, order=3)
    
    # Step 2: Apply Bandpass Filter
    # UPDATED: Adjusted highcut to 0.6Hz to capture more signal morphology
    sig_bandpass = bandpass_filter(sig_no_wander, 0.05, 0.6, fs)
    
    # Step 3: Normalize
    final_signal = normalize_signal(sig_bandpass, method='minmax')
    
    print("  [INFO] Signal cleaned and normalized with optimized parameters.")
    return final_signal
