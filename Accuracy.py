import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from transforms import AddPinkNoise, DropInput, AddGaussianNoise, GaussianCorruption, TwoCropTransform
from dataset import Motion_Dataset_Patient
from models import WaveFusion_Feature_Model, WaveFusion_contrastive_classifier_Model
from collections import defaultdict

# Define your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your transforms
val_transform = torch.nn.Sequential(
    AddPinkNoise(mean=0, std=0.85, p=0.9, device_=device),
    DropInput(p=0.4, x=0.5, device_=device),
    GaussianCorruption(mean=0, std=0.02, p=0.5, x=0.5, device_=device),
)

# Load your validation data
data_dir = r"C:\Users\juulv\Desktop\Documents\Universiteit\universiteit 2023 - 2024\Thesis\data\32Confidence"  # Define the data directory
batch_size = 4000  # Define the batch size

motion_datasets = {x: Motion_Dataset_Patient(os.path.join(data_dir, x), return_truths=True, transform=val_transform if x == 'val' else None) for x in ['train', 'val']}
val_loader = DataLoader(motion_datasets['val'], batch_size=batch_size, shuffle=True)

save_path = r"C:\Users\juulv\Desktop\Documents\Universiteit\universiteit 2023 - 2024\Thesis\Model_extract\confidence\run2"
fname = '_attention_weightsWaveFusion_Param_0.0_batchSize500_ACC0.9992499947547913_embeWd0.001_supConTemp0.1_attnTemp27.5_clfDp0.67_clfWd0.005'
model_path = os.path.join(save_path, fname)

# Initialize new model
patient_model = WaveFusion_contrastive_classifier_Model(device=device).to(device)

# Load pretrained weights to wavefusion feature extractor
def load_weights(self, model_path):
    try:
        weight_dict = torch.load(model_path)
        new_state_dict = {k: v for k, v in zip(self.state_dict().keys(), weight_dict.values())}
        self.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")

load_weights(patient_model, model_path)

# Make predictions on validation data
all_predictions = []
all_labels = []
all_truths = []
subject_data = defaultdict(lambda: {"predictions": [], "truths": []})

# Initialize dictionaries to store accuracies and F1 scores
subject_accuracy_truths = {}
subject_accuracy_labels = {}
subject_f1_labels = {}
subject_class_accuracies = {}
subject_sensitivity = {}


# Flag to determine if metacognition is based on sensitivity
metacognition_is_sensitivity = True  # Set this flag based on your requirement

try:
    for data, labels, truths, subject_id in val_loader:    
        outputs = patient_model(data)[0]
        _, predictions = torch.max(outputs, 1)

        # Map predictions to [0, 1] and switch order
        predictions = 1 - (predictions > 0.5).long().cpu().numpy()  # For sensitivity model predictions these are inverted
        truths = truths.cpu().numpy()
        labels = labels.cpu().numpy()  # Ensure labels are in the correct format
        subject_ids = subject_id.cpu().numpy()
        labels = labels[:, 0]  # Extract the first column of the labels (as they were extracted as the key)

        accuracy_truths = accuracy_score(truths, predictions)
        accuracy_labels = accuracy_score(labels, predictions)

        for single_subject_id in np.unique(subject_ids):
            subject_mask = (subject_ids == single_subject_id).flatten()
            subject_predictions = predictions[subject_mask]
            subject_truths = truths[subject_mask]
            subject_labels = 1 - labels[subject_mask]

            subject_accuracy_truths[single_subject_id] = accuracy_score(subject_truths, subject_predictions)
            subject_accuracy_labels[single_subject_id] = accuracy_score(subject_labels, subject_predictions)
            subject_f1_labels[single_subject_id] = f1_score(subject_labels, subject_predictions)


            class_0_mask = (subject_labels == 0)
            class_1_mask = (subject_labels == 1)
            class_0_accuracy = accuracy_score(subject_labels[class_0_mask], subject_predictions[class_0_mask])
            class_1_accuracy = accuracy_score(subject_labels[class_1_mask], subject_predictions[class_1_mask])
            subject_class_accuracies[single_subject_id] = {
                'class_0_accuracy': class_0_accuracy,
                'class_1_accuracy': class_1_accuracy
            }

            if metacognition_is_sensitivity:
                # Calculate overall accuracy for class 0
                correct_class_0_predictions = np.count_nonzero(subject_predictions == 0) # Count the number of correct class 0 predictions
                print(correct_class_0_predictions)
                total_class_0_predictions = len(subject_predictions)
                overall_accuracy_class_0 = correct_class_0_predictions / total_class_0_predictions # Calculate the overall accuracy for class 0, the metacognitive sensitivty
                subject_sensitivity[single_subject_id] = overall_accuracy_class_0

    # Calculate total scores
    total_accuracy_truths = sum(subject_accuracy_truths.values()) / len(subject_accuracy_truths)
    total_accuracy_labels = sum(subject_accuracy_labels.values()) / len(subject_accuracy_labels)
    total_f1_labels = sum(subject_f1_labels.values()) / len(subject_f1_labels)
    total_class_0_accuracy = sum(class_accuracies['class_0_accuracy'] for class_accuracies in subject_class_accuracies.values()) / len(subject_class_accuracies)
    total_class_1_accuracy = sum(class_accuracies['class_1_accuracy'] for class_accuracies in subject_class_accuracies.values()) / len(subject_class_accuracies)


    output_file_path = os.path.join(save_path, 'accuracies_output.txt')

    # Open a text file to write the accuracies and F1 scores
    with open(output_file_path, 'w') as file:
        file.write("Accuracies based on truths:\n")
        for subject_id, accuracy in subject_accuracy_truths.items():
            file.write(f"Accuracy for subject {subject_id} based on truths: {accuracy}\n")

        file.write("\nAccuracies based on labels:\n")
        for subject_id, accuracy in subject_accuracy_labels.items():
            file.write(f"Accuracy for subject {subject_id} based on labels: {accuracy}\n")

        file.write("\nF1 scores based on labels:\n")
        for subject_id, f1 in subject_f1_labels.items():
            file.write(f"F1 score for subject {subject_id} based on labels: {f1}\n")

        file.write("\nClass-wise accuracies based on labels:\n")
        for subject_id, class_accuracies in subject_class_accuracies.items():
            file.write(f"Class 0 accuracy for subject {subject_id}: {class_accuracies['class_0_accuracy']}\n")
            file.write(f"Class 1 accuracy for subject {subject_id}: {class_accuracies['class_1_accuracy']}\n")

        file.write("\nTotal scores for all subjects combined:\n")
        file.write(f"Total accuracy based on truths: {total_accuracy_truths}\n")
        file.write(f"Total accuracy based on labels: {total_accuracy_labels}\n")
        file.write(f"Total F1 score based on labels: {total_f1_labels}\n")
        file.write(f"Total class 0 accuracy: {total_class_0_accuracy}\n")
        file.write(f"Total class 1 accuracy: {total_class_1_accuracy}\n")


        if metacognition_is_sensitivity:
            file.write(f"Total sensitivity for class 0: {subject_sensitivity}")

    for subject_id, sensitivity in subject_sensitivity.items():
        file.write(f"Sensitivity for subject {subject_id}: {sensitivity}\n")

    print("File written successfully.")
except Exception as e:
    print(f"Error during processing: {e}")


