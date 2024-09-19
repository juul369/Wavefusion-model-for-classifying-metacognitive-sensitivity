import numpy as np
from scipy import signal
from scipy.io import loadmat
import os
import random
from sklearn.model_selection import train_test_split

# Define the Hann window
window = signal.windows.hann(80)
def generate_averaged_recordings(eeg_data, truths, num_recordings, percentage_to_average, sensitivity):
    """
    Generate averaged EEG recordings.
    
    Parameters:
    - eeg_data: List of EEG recordings.
    - truths: List of truth values corresponding to the EEG recordings.
    - num_recordings: Number of averaged recordings to generate.
    - percentage_to_average: Percentage of total samples to average over for each recording.
    - sensitivity: Boolean indicating if the function is being used for sensitivity analysis.
    
    Returns:
    - A list of averaged EEG recordings.
    - A list of rounded truth values if sensitivity is False.
    """
    averaged_recordings = []
    rounded_truths = []
    
    total_samples = len(eeg_data)
    samples_to_average = int(total_samples * percentage_to_average)
    
    for _ in range(num_recordings):
        # Randomly select samples with replacement
        selected_indices = random.choices(range(total_samples), k=samples_to_average)
        averaged_recording = np.mean([eeg_data[i] for i in selected_indices], axis=0)
        
        averaged_recordings.append(averaged_recording)
        
        if not sensitivity:
            averaged_truth = round(np.mean([truths[i] for i in selected_indices]))
            rounded_truths.append(averaged_truth)
    
    if sensitivity:
        return averaged_recordings
    else:
        return averaged_recordings, rounded_truths
            


# Iterate over the subjects
for subject in range(1, 17):
    # Load the EEG data
    mat = loadmat(f"C:\\Users\\juulv\\Desktop\\Documents\\Universiteit\\universiteit 2023 - 2024\\Thesis\\Code\\Data ERP boldtYeung\\confeeg03_f20_S{subject}.mat")
    eeg_data = mat['data']
    confidence_label = mat['cj'][0]
    truths = mat['truths'][0] # 0 = correct, 1 = incorrect
    print(eeg_data.shape)

    # Downsample the data from 1000Hz to 100Hz
    downsampled_eeg = signal.decimate(eeg_data, 10, axis=1)

    # Initialize lists to store high and low confidence EEG data and their truths
    high_confidence_eeg = []
    low_confidence_eeg = []
    high_confidence_truths = []
    low_confidence_truths = []

    High_confidence_correct_eeg = []
    High_confidence_incorrect_eeg = []
    Low_confidence_correct_eeg = []
    Low_confidence_incorrect_eeg = []

    # choose method
    sensitivity = False # True = split data on sensitivity, False = split data on confidence

    # Split the EEG data based on confidence and keep truths aligned
    for i, label in enumerate(confidence_label):
        if label > 3:
            high_confidence_eeg.append(downsampled_eeg[:, :, i])
            high_confidence_truths.append(truths[i])
            if truths[i] == 0:
                High_confidence_correct_eeg.append(downsampled_eeg[:, :, i])
            else:
                High_confidence_incorrect_eeg.append(downsampled_eeg[:, :, i])
        else:
            low_confidence_eeg.append(downsampled_eeg[:, :, i])
            low_confidence_truths.append(truths[i])
            if truths[i] == 1: # 1 = correct because low confidence judgement is judging that it's incorrect, thus a correct judgement
                Low_confidence_correct_eeg.append(downsampled_eeg[:, :, i])
            else:
                Low_confidence_incorrect_eeg.append(downsampled_eeg[:, :, i])


    # Store the number of high and low confidence correct and incorrect EEG recordings per subject
    subject_data = {
    "High_confidence_correct": len(High_confidence_correct_eeg),
    "High_confidence_incorrect": len(High_confidence_incorrect_eeg),
    "Low_confidence_correct": len(Low_confidence_correct_eeg),
    "Low_confidence_incorrect": len(Low_confidence_incorrect_eeg)
    }

    ### SENSITIVITY
    if sensitivity == True:
        # Combine high and low confidence correct and incorrect EEG data into single variables
        correct_eeg = High_confidence_correct_eeg + Low_confidence_correct_eeg
        incorrect_eeg = High_confidence_incorrect_eeg + Low_confidence_incorrect_eeg

        # Optionally, convert these combined lists to numpy arrays if needed for further processing
        correct_eeg = np.array(correct_eeg)
        incorrect_eeg = np.array(incorrect_eeg)

        # Shuffle the correct and incorrect trials
        np.random.shuffle(correct_eeg)
        np.random.shuffle(incorrect_eeg)

        # Split data and truths into train and validation sets for correct and incorrect EEG data
        correct_train, correct_val = train_test_split(correct_eeg, test_size=0.5, random_state=42)
        incorrect_train, incorrect_val = train_test_split(incorrect_eeg, test_size=0.5, random_state=42)

        print("Sizes of all split EEG data:")
        print(f"Correct sensitivity EEG data - Train: {correct_train.shape}, Val: {correct_val.shape}")
        print(f"Incorrect sensitivity EEG data - Train: {incorrect_train.shape}, Val: {incorrect_val.shape}")

        # Generate augmented data for correct and incorrect sensitivity EEG recordings
        correct_train_augmented = generate_averaged_recordings(correct_train, [], 500, 0.1,sensitivity)
        correct_val_augmented = generate_averaged_recordings(correct_val, [],250, 0.1,sensitivity)

        incorrect_train_augmented = generate_averaged_recordings(incorrect_train,[], 500, 0.1,sensitivity)
        incorrect_val_augmented = generate_averaged_recordings(incorrect_val,[], 250, 0.1,sensitivity)

        print("Sizes of augmented EEG data and truths arrays:")
        print(f"Correct sensitivity - Train: Data {len(correct_train_augmented)}")
        print(f"Correct sensitivity - Val: Data {len(correct_val_augmented)}")
        print(f"Incorrect sensitivity - Train: Data {len(incorrect_train_augmented)}")
        print(f"Incorrect sensitivity - Val: Data {len(incorrect_val_augmented)}")

        # Apply the STFT and store results with distinct variable names
        frequencies_correct_train, times_correct_train, spectrogram_correct_train = signal.stft(correct_train_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)
        frequencies_correct_val, times_correct_val, spectrogram_correct_val = signal.stft(correct_val_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)
        frequencies_incorrect_train, times_incorrect_train, spectrogram_incorrect_train = signal.stft(incorrect_train_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)
        frequencies_incorrect_val, times_incorrect_val, spectrogram_incorrect_val = signal.stft(incorrect_val_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)

        # Take the absolute value to get the magnitude for each spectrogram
        spectrogram_correct_train = np.abs(spectrogram_correct_train)
        spectrogram_correct_val = np.abs(spectrogram_correct_val)
        spectrogram_incorrect_train = np.abs(spectrogram_incorrect_train)
        spectrogram_incorrect_val = np.abs(spectrogram_incorrect_val)


        # Select the first 12 leads from each spectrogram
        spectrogram_correct_train = spectrogram_correct_train[:,:-16, :, :]
        spectrogram_correct_val = spectrogram_correct_val[:,:-16, :, :]
        spectrogram_incorrect_train = spectrogram_incorrect_train[:,:-16, :, :]
        spectrogram_incorrect_val = spectrogram_incorrect_val[:,:-16, :, :]

        print("Sizes of all spectrogram arrays:")
        print(f"Correct sensitivity - Train: {spectrogram_correct_train.shape}, Val: {spectrogram_correct_val.shape}")


        # Create directories if they don't exist
        if not os.path.exists('train/correct_group'):
            os.makedirs('train/correct_group')
        if not os.path.exists('train/incorrect_group'):
            os.makedirs('train/incorrect_group')
        if not os.path.exists('val/correct_group'):
            os.makedirs('val/correct_group')
        if not os.path.exists('val/incorrect_group'):
            os.makedirs('val/incorrect_group')

        # Save spectrograms for training data with augmented truths
        for i in range(len(spectrogram_correct_train)):
            label = 1  # label 1 correct sensitivity in dataset.py it's converted to a 0
            truth = 0  #  correct group
            group = 'correct_group'
            np.save(f'train/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_correct_train[i])

        for i in range(len(spectrogram_incorrect_train)):
            label = 4  # label 4 for incorrect sensitivity in dataset.py it's converted to a 1
            truth = 1  # Updated to use truths_augmented for incorrect group
            group = 'incorrect_group'
            np.save(f'train/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_incorrect_train[i])

        # Save spectrograms for validation data with augmented truths
        for i in range(len(spectrogram_correct_val)):
            label = 1  # label 1 for correct sensitivity
            truth = 0  # Updated to use truths_augmented for correct group
            group = 'correct_group'
            np.save(f'val/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_correct_val[i])

        for i in range(len(spectrogram_incorrect_val)):
            label = 4  # label 4 for incorrect sensitivity in dataset.py it's converted to a 1
            truth = 1  # Updated to use truths_augmented for incorrect group
            group = 'incorrect_group'
            np.save(f'val/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_incorrect_val[i])


    else:
    ### CONFIDENCE
        # Convert lists to numpy arrays before augmentation
        high_confidence_eeg = np.array(high_confidence_eeg)
        low_confidence_eeg = np.array(low_confidence_eeg)
        high_confidence_truths = np.array(high_confidence_truths)
        low_confidence_truths = np.array(low_confidence_truths)

        # Function to split data and truths into train and validation sets
        def split_data_and_truths(data, truths, val_size=0.3, random_state=42):
            data_train, data_val, truths_train, truths_val = train_test_split(data, truths, test_size=val_size, random_state=random_state)
            return data_train, data_val, truths_train, truths_val

        # Split high confidence EEG data and truths
        high_train, high_val, high_train_truths, high_val_truths = split_data_and_truths(high_confidence_eeg, high_confidence_truths)

        # Split low confidence EEG data and truths
        low_train, low_val, low_train_truths, low_val_truths = split_data_and_truths(low_confidence_eeg, low_confidence_truths)

        print("Sizes of all truths arrays:")
        print(f"High confidence truths - Train: {high_train_truths.shape[0]}, Val: {high_val_truths.shape[0]}")
        print(f"Low confidence truths - Train: {low_train_truths.shape[0]}, Val: {low_val_truths.shape[0]}")

        print("Sizes of all split EEG data:")
        print(f"High confidence EEG data - Train: {high_train.shape}, Val: {high_val.shape}")
        print(f"Low confidence EEG data - Train: {low_train.shape}, Val: {low_val.shape}")

        #### averaging augmentation

        # Generate augmented data for high confidence EEG recordings
        high_train_augmented, high_train_truths_augmented = generate_averaged_recordings(high_train, high_train_truths, 500, 0.1, sensitivity)
        high_val_augmented, high_val_truths_augmented = generate_averaged_recordings(high_val, high_val_truths, 250, 0.1, sensitivity)

        # Generate augmented data for low confidence EEG recordings
        low_train_augmented, low_train_truths_augmented = generate_averaged_recordings(low_train, low_train_truths, 500, 0.1, sensitivity)
        low_val_augmented, low_val_truths_augmented = generate_averaged_recordings(low_val, low_val_truths, 250, 0.1, sensitivity)

        print("Sizes of augmented EEG data and truths arrays:")
        print(f"High confidence - Train: Data {len(high_train_augmented)}, Truths {len(high_train_truths_augmented)}")
        print(f"High confidence - Val: Data {len(high_val_augmented)}, Truths {len(high_val_truths_augmented)}")
        print(f"Low confidence - Train: Data {len(low_train_augmented)}, Truths {len(low_train_truths_augmented)}")
        print(f"Low confidence - Val: Data {len(low_val_augmented)}, Truths {len(low_val_truths_augmented)}")
    
        
        # Apply the STFT
        # Apply the STFT and store results with distinct variable names
        frequencies_high_train, times_high_train, spectrogram_high_train = signal.stft(high_train_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)
        frequencies_high_val, times_high_val, spectrogram_high_val = signal.stft(high_val_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)
        frequencies_low_train, times_low_train, spectrogram_low_train = signal.stft(low_train_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)
        frequencies_low_val, times_low_val, spectrogram_low_val = signal.stft(low_val_augmented, fs=100, window=window, noverlap=int(0.75 * 80), nperseg=80, axis=2)
        
        # Take the absolute value to get the magnitude for each spectrogram
        spectrogram_high_train = np.abs(spectrogram_high_train)
        spectrogram_high_val = np.abs(spectrogram_high_val)
        spectrogram_low_train = np.abs(spectrogram_low_train)
        spectrogram_low_val = np.abs(spectrogram_low_val)

        # Select the first 12 leads from each spectrogram
        spectrogram_high_train = spectrogram_high_train[:, :-16, :, :]
        spectrogram_high_val = spectrogram_high_val[:, :-16, :, :]
        spectrogram_low_train = spectrogram_low_train[:, :-16, :, :]
        spectrogram_low_val = spectrogram_low_val[:, :-16, :, :]

        # Create directories if they don't exist
        if not os.path.exists('train/high_group'):
            os.makedirs('train/high_group')
        if not os.path.exists('train/low_group'):
            os.makedirs('train/low_group')
        if not os.path.exists('val/high_group'):
            os.makedirs('val/high_group')
        if not os.path.exists('val/low_group'):
            os.makedirs('val/low_group')

            # Save spectrograms for training data with augmented truths
        for i in range(len(spectrogram_high_train)):
            label = 4
            truth = high_train_truths_augmented[i]  # Updated to use truths_augmented for high group
            group = 'high_group'
            np.save(f'train/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_high_train[i])

        for i in range(len(spectrogram_low_train)):
            label = 1
            truth = low_train_truths_augmented[i]  # Updated to use truths_augmented for low group
            group = 'low_group'
            np.save(f'train/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_low_train[i])

        # Save spectrograms for validation data with augmented truths
        for i in range(len(spectrogram_high_val)):
            label = 4
            truth = high_val_truths_augmented[i]  # Updated to use truths_augmented for high group
            group = 'high_group'
            np.save(f'val/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_high_val[i])

        for i in range(len(spectrogram_low_val)):
            label = 1
            truth = low_val_truths_augmented[i]  # Updated to use truths_augmented for low group
            group = 'low_group'
            np.save(f'val/{group}/spectrogram_{subject}_{i}_{label}_{truth}.npy', spectrogram_low_val[i])


    # Write the values to a single text file
    output_file = "all_subjects_data.txt"
    with open(output_file, "a") as file:  # Use "a" to append to the file
        file.write(f"Subject: {subject}\n")
        for key, value in subject_data.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")  # Add a newline for separation between subjects


