import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt, find_peaks, welch

def load_data_from_csv(csv_path, row_index=0, fs=125):
    """
    Loads PPG signal and ground truth respiratory rate from a CSV file.
    """
    # Check if the file exists to prevent crash
    if not os.path.exists(csv_path):
        print(f"ERROR: File '{csv_path}' not found!")
        return None, None, None

    try:
        # Load the dataset (assuming first row is header)
        df = pd.read_csv(csv_path, header=0)
        
        # Extract the specific row for the selected sample
        data_row = df.iloc[row_index].values
        
        # First 4000 columns contain the PPG signal data
        ppg_signal = data_row[:4000].astype(float)
        
        # The last column contains the Ground Truth (Respiratory Rate)
        ground_truth = float(data_row[-1])
        
        # Generate time axis based on sampling frequency (fs)
        t = np.arange(len(ppg_signal)) / fs
        
        return t, ppg_signal, ground_truth
        
    except Exception as e:
        # Handle errors (e.g., wrong format, missing columns)
        print(f"Error reading data: {e}")
        return None, None, None

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

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies Butterworth Bandpass filter to isolate the relevant frequency band.
    Uses SOS (second-order sections) format for numerical stability with 
    low frequency cutoffs and high sampling rates.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Safety checks for digital filter boundaries
    if low <= 0: low = 0.001
    if high >= 1: high = 0.999
    
    # Use SOS format for numerical stability
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, data)

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

def estimate_rr_time_domain(signal, fs):
  
    """
    Estimates the respiratory rate (RR) in Breaths Per Minute (BPM) 
    from a respiratory signal using a time-domain peak-detection approach.

    Args:
        signal (np.array): The respiratory signal (e.g., from an accelerometer or respiration belt).
        fs (float): The sampling frequency of the signal in Hz.

    Returns:
        tuple: (rr_bpm, peaks) where rr_bpm is the estimated respiratory rate 
               and peaks is an array of detected peak indices.
    """
    
    # Calculate the minimum required distance between successive peaks in samples.
    # Assuming a minimum physiological breathing rate (e.g., 40 breaths/min, period of 1.5s).
    # This prevents detecting multiple peaks for a single breath cycle.
    peaks, _ = find_peaks(signal, distance=fs*1.5)
    
    # Goal: To ensure that the find_peaks function detects only one peak representing each actual breath cycle (inhalation/exhalation), 
    # instead of detecting small oscillations caused by noise within a single respiratory cycle.
    #40 BPM (40 breaths per minute) corresponds to approximately a 1.5-second period. 
    #The use of the distance 1.5 * {fs} in respiratory rate (RR) estimation is based on this.
    num_peaks = len(peaks)
    
    # Calculate the total duration of the recorded signal in seconds.
    duration_sec = len(signal) / fs

    # Estimate the respiratory rate in BPM using the formula: 
    # RR (BPM) = (Number of Peaks / Total Duration in seconds) * 60 seconds/minute.
    rr_bpm = (num_peaks / duration_sec) * 60
    
    # Return the estimated respiratory rate and the indices of the detected peaks.
    return rr_bpm, peaks

def estimate_rr_freq_domain(signal, fs):

    # Compute power spectral density using Welch's method
    f, Pxx = welch(signal, fs, nperseg=1024)

    # Find the index of the dominant frequency
    peak_freq_index = np.argmax(Pxx)

    # Get the dominant frequency value
    peak_freq = f[peak_freq_index]

    # Convert frequency from Hz to breaths per minute (BPM)
    rr_bpm = peak_freq * 60

    return rr_bpm, f, Pxx

def plot_respiratory_analysis(time, raw_signal, clean_signal, peaks, rr_time, freqs, psd, rr_freq):

    # Setting up the figure size for the entire plot
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Raw Sgnal
    plt.subplot(4, 1, 1)
    # Plotting the original, unprocessed data with gray color
    plt.plot(time, raw_signal, color='gray', alpha=0.6, label='Raw Data')
    plt.title('Step 1: Raw PPG Signal (From Dataset)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Filtered Signal
    plt.subplot(4, 1, 2)
    # Plot the signal after bandpass filtering (the making of the 'clean' signal)
    plt.plot(time, clean_signal, color='blue', label='Processed Signal')
    plt.title('Step 2: Filtered and Normalized Signal (0.1-0.5 Hz)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Time Domain Analysis and Peak Detection
    plt.subplot(4, 1, 3)
    # Plot the filtered signal again
    plt.plot(time, clean_signal, color='green', alpha=0.6)
    # Mark the detected peaks (representing 'breaths') with red 'x' markers
    plt.plot(time[peaks], clean_signal[peaks], "x", color='red', markersize=8, markeredgewidth=2, label='Breaths')
    # Display the respiratory rate (RR) estimate calculated in the time domain
    plt.title(f'Step 3: Time Domain Analysis (Estimate: {rr_time:.2f} BPM)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Frequency Domain Analysis
    plt.subplot(4, 1, 4)
    # Plot the Power Spectral Density (PSD) using a semi-log scale for power
    plt.semilogy(freqs, psd, color='purple')
    # Draw a vertical line to highlight the peak frequency that determines RR
    plt.axvline(x=rr_freq/60, color='red', linestyle='--', linewidth=2, label=f'Peak: {rr_freq/60:.2f} Hz')
    # Set the x-axis limit to focus on the respiratory frequency range
    plt.xlim(0, 1)
    # Display the respiratory rate (RR) estimate calculated in the frequency domain
    plt.title(f'Step 4: Frequency Domain Analysis (Estimate: {rr_freq:.2f} BPM)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (Log)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "BIDMC32RR_TRAIN.csv")   
    sample_fs = 125         
    row_to_analyze = 25   

    print("--- Respiratory Rate Estimation Project ---")

    time_axis, raw_ppg_signal, ground_truth_rr = load_data_from_csv(csv_path, row_index=row_to_analyze, fs=sample_fs)

    if time_axis is not None:
        print(f"Dataset Loaded. Ground Truth RR: {ground_truth_rr} BPM")

        clean_ppg_signal = preprocess_signal_pipeline(raw_ppg_signal, sample_fs)

        rr_est_time, peak_indices = estimate_rr_time_domain(clean_ppg_signal, sample_fs)
        print(f"Time Domain Estimate: {rr_est_time:.2f} BPM")

        rr_est_freq, freq_axis, psd_values = estimate_rr_freq_domain(clean_ppg_signal, sample_fs)
        print(f"Frequency Domain Estimate: {rr_est_freq:.2f} BPM")

        plot_respiratory_analysis(
            time=time_axis,
            raw_signal=raw_ppg_signal,
            clean_signal=clean_ppg_signal,
            peaks=peak_indices,
            rr_time=rr_est_time,
            freqs=freq_axis,
            psd=psd_values,
            rr_freq=rr_est_freq
        )
    else:
        print("Data could not be loaded. Please check the 'csv_path'.")