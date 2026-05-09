'''This code defines a visualization function, plot_respiratory_analysis, designed to illustrate the entire signal processing work line for extracting Respiratory Rate (RR) from a raw signal, typically Photoplethysmography (PPG) data. It generates a four-panel figure that sequentially displays:

    1-Raw Signal: Has the unprocessed input data.
    2-Filtered Signal: Makes the signal after bandpass filtering (0.1-0.5 Hz) to isolate the respiratory component.
    3-Time Domain Analysis: Shows the clean signal with little 'x' marks on the peaks (the breaths), then uses the time between those peaks to figure out the RR in Breaths Per Minute (BPM).
    4-Frequency Domain Analysis: Plots the Power Spectral Density (PSD), which shows how much each frequency contributes. We look for the tallest peak, and that peak tells us the most likely RR, giving us another estimate in BPM.'''


import matplotlib.pyplot as plt
import numpy as np

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
