import pandas as pd
import numpy as np
import os

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