import numpy as np
from scipy.signal import welch

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
