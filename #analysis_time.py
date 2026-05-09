#analysis_time.py

from scipy.signal import find_peaks

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