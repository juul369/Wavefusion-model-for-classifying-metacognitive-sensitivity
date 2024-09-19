import torch
import os
import pandas as pd
import numpy as np

# Define the path to the attention weights
save_path = r"C:\Users\juulv\Desktop\Documents\Universiteit\universiteit 2023 - 2024\Thesis\Model_extract\confidence\run2"
fname = '_attention_weightsWaveFusion_Param_0.0_batchSize500_ACC0.9992499947547913_embeWd0.001_supConTemp0.1_attnTemp27.5_clfDp0.67_clfWd0.005'

# Load the state dictionary
attention_weights = state_dict = torch.load(os.path.join(save_path, fname))

# Convert the attention weights to a numpy array and flatten it
attention_weights_np = attention_weights.numpy().flatten()

# Print the attention weights
print("Attention Weights:", attention_weights_np.shape)

# Find the minimum value in the attention weights: to create an all positive topoplot, the weights only indicate to which areas more attention is paid
min_value = np.min(attention_weights_np)

# Subtract the minimum value from all the attention weights to set it as the baseline
attention_weights_baseline_shifted = attention_weights_np - min_value

# Optional: Scale the shifted weights
# You can adjust the scaling factor to make the difference more visible
scaling_factor = 1  # Adjust this factor as needed
attention_weights_scaled = attention_weights_baseline_shifted * scaling_factor



# Print the shifted and optionally scaled attention weights
print("Attention Weights with baseline shifted:", attention_weights_scaled)

# Convert the numpy array to a DataFrame
attention_weights_df = pd.DataFrame(attention_weights_scaled)

# Set print options to avoid truncation
np.set_printoptions(threshold=np.inf)

# Write the attention weights to a text file
output_file_path = os.path.join(save_path, "attention_weights.txt")
with open(output_file_path, "w") as file:
    file.write("Attention Weights before baseline correction:\n")
    file.write(np.array2string(attention_weights_np, separator=', ') + "\n\n")
    file.write("Attention Weights after baseline correction:\n")
    file.write(np.array2string(attention_weights_scaled, separator=', ') + "\n")


# Save the DataFrame to a CSV file
# attention_weights_df.to_csv("attention_weights.csv")
import numpy as np
import matplotlib.pyplot as plt
import mne

# attention_weights_np is correctly ordered according to electrode_names
electrode_names = [ 'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'C3', 'Cz', 
                   'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7','P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2' ]
# frontal: 'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'C3', 'Cz'
# posterior: 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7','P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'


# Create a standard montage from MNE
montage = mne.channels.make_standard_montage('standard_1020')

# Get positions from the montage, only include electrodes present in the montage
positions = {ch_name: montage.get_positions()['ch_pos'][ch_name] for ch_name in electrode_names if ch_name in montage.get_positions()['ch_pos']}

# Create a custom montage (this step is crucial for plotting a subset of electrodes effectively)
montage_custom = mne.channels.make_dig_montage(ch_pos=positions, coord_frame='head')

# Create a fake Info object, necessary for plotting with MNE
info = mne.create_info(ch_names=electrode_names, sfreq=256, ch_types="eeg")
info.set_montage(montage_custom)

# Now, use MNE's plotting function, ensuring pos is derived from the custom montage
fig, ax = plt.subplots()
mne.viz.plot_topomap(attention_weights_scaled, info, sensors=True, names=electrode_names, axes=ax, show=False, sphere=0.115) #  

plt.show()