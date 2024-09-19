import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import BatchSampler, RandomSampler
import os, sys
from typing import cast, Tuple

import math
import numpy as np
class Motion_Dataset_Patient(Dataset):

    def __init__(self, root_dir, threeD = False, transform=None, device = None, return_truths=False, sensitivity = False):
        """
        Args:
            direction (bool): True if classifying direction. Otherwise classify confidence.
            root_dir (string): root directory with subdirectory containing .npy files of eeg readings class 1-4.
            threeD (boolean): true if doing 3D convolution
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        direction = False
        if direction:
            class_to_idx = {'90': 0, '270': 1}
        else:
            class_to_idx = {'1': 0,'4': 1}
        self.extensions = ".npy"
        self.transform = transform
        self.threeD = threeD
        self.num_chans = 32
        self.device = device
        self.return_truths = return_truths # eventually did not use this feature
        #self.sensitivity = sensitivity

        self.samples = np.array(self.make_dataset(root_dir, self.extensions, class_to_idx))
        self.subject_id = self.patient_dictionary(self.samples)

        self.the_data = np.empty((len(self.samples), 32,41,10))
        self.targets = np.empty((len(self.samples),2))
        self.truths = np.empty((len(self.samples),1))
        self.subject_ids = np.empty((len(self.samples), 1))

        
        for i, sample in  enumerate(self.samples): ## modified for nice SDT output: get all tensors for the data, targets, truths and subject_ids, with correpsonding positions
            self.the_data[i,:,:,:] = np.load(sample[0], allow_pickle = True)
            self.targets[i] = int(sample[1]), int(self.integer_encode_target(sample[0], self.subject_id )) # (1,0) is high confidence key, (0,1) is low confidence key
            self.truths[i] = self.integer_encode_truth(sample[0])
            self.subject_ids[i] = self.integer_encode_ID(sample[0])


            #print(self.the_data.shape)
            #print(self.targets.shape)
            #print(self.truths.shape)
            #print("truths", self.truths[i])
            #print("targets", self.targets[i])

            # Metacognitive sensitivity calculaton, by changing confidence to (1,0) if correct (high = correct, low = incorrect) or (0,1) if incorrect (high = incorrect, low = correct), only  operations required
            #if self.sensitivity:
               #if np.array_equal(self.targets[i], [1, 0]) and self.truths[i] == 1:
                    #self.targets[i] = [0, 1]
               #if np.array_equal(self.targets[i], [0, 1]) and self.truths[i] == 0:
                    #self.targets[i] = [1.0, 0.0]
            # self. targets now represent metacognitive sensitivity instead of confidence
                

    def has_file_allowed_extension(self, filename, extensions):
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """

        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)

    def is_valid_file(self, x: str):
            return self.has_file_allowed_extension(x, cast(Tuple[str, ...], self.extensions))

    def make_dataset(self, root_dir, extensions, class_to_idx):
        """creates a list of paths to data in root data directory

        Args:
            root_dir (string): root directory with subdirectory containing .npy files of eeg readings class 1-6.
            extensions: extensions allowed.
            class_to_idx: dictionary of classes and numerics.

        Returns:
            instances (list): list of tuples with path to .npy file and target.
        """

        instances = []

        for root, _, fnames in sorted(os.walk(root_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if self.is_valid_file(path):
                    class_index = class_to_idx[fname.split('_')[3][0]]
                    item = path, class_index
                    instances.append(item)
        return instances


    def integer_encode_target(self,sample,subject_id):
        tmp = sample.split("/")[-1].split("_")
        return self.subject_id[tmp[0]]
    
    ## new piece of code
    def integer_encode_truth(self, sample):
        # Split the sample name on the "_" character
        tmp = sample.split('_')

        # Get the truth from the desired position in the name
        truth = tmp[5]  # Adjust this index to match the position of the truth in your sample names
        truth = truth.replace('.npy', '')

        # Return the truth as an integer
        return int(truth)
    
    ## new piece of code
    def integer_encode_ID(self, sample):
        # Split the sample name on the "_" character
        with open('SampleID order.txt', 'a') as f:
            f.write(str(os.path.basename(sample).split('_')) + '\n')
        subjectID = os.path.basename(sample).strip().split('_')[1]
        
        # Return the truth as an integer
        return int(subjectID)

    def patient_dictionary(self, samples):
        subject_id = {}
        id = 0
        for i in range(len(samples)):
            tmp = samples[i][0].split("/")[-1].split("_")
            key = tmp[0]
            if key not in subject_id.keys():
                subject_id[key] = id
                id += 1
        return subject_id

    def __getitem__(self, indexes: int): # changed this to include truths and subject ID (for SDT code)!!!!!!!! change again to just give the tensors?

            data = torch.Tensor( self.the_data[indexes,:,:,:])
            targets = torch.LongTensor(self.targets[indexes] )
            if self.return_truths:
                truths = torch.Tensor(self.truths[indexes])
                subject_ids = torch.LongTensor(self.subject_ids[indexes])
                return data, targets, truths, subject_ids
            else:
                return data, targets

    def __len__(self):
        return len(self.samples)

# In[12]:
def motion_collate(batch):
    """
    custom collate function for turning a batch of scale-time wavelet transforms types into
    (batch_size)x17x48x2000 TENSORS with target

    Args:
        Batch: batch of Lemon STFT Data

    Returns:
        tuple: (tesseractTransform(sample(index)), target) where target is class_index of the target class.
    """
    return batch

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, num_pat_spilt = (525,100,425) ):

        #get all labels in dataset
        self.labels = dataset.targets
        self.num_pat_spilt = num_pat_spilt
        self.labels_dict = self.labels_dictionary(self.labels)
        
        #get all unique labels
        self.labels_set = np.array(list(self.labels_dict.values()))

        self.idx_labels = np.array([self.labels_dict[ (int(label[0]), int(label[1])) ] for label in self.labels ])


        #array of indices for each class.patient label
        self.label_to_indices = {label: np.where(self.idx_labels == label)[0]
                                 for label in self.labels_set}
        count = 0
        for l in self.label_to_indices:
            count += len(self.label_to_indices[l])

        # shuffle indices in label_to_indices
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        
        #keeps track of the number of labels used?
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.n_classes = len(self.labels_set)
        self.dataset = dataset
        self.dataset_size = len(self.dataset)
        self.counter = 0
        self.batch_size = self.num_pat_spilt[0]
        self.num_pos_samples = self.num_pat_spilt[1]
        self.num_neg_samples = self.num_pat_spilt[2]
        self.num_iters = int(self.dataset_size / self.num_pos_samples)

        # made a change here, added a condition to check if there are only two classes, otherwise it would've given a zerodeiviosn error
            # previous code assumed there must be more than 2 classes
        if len(self.labels_set) <= 2:
            self.num_neg_others = 0
        else:
            self.num_neg_others = int( (self.batch_size-self.num_pos_samples-self.num_neg_samples) / ((len(self.labels_set)-2)/2) )

    def labels_dictionary(self, labels):
        labels_id = {}
        idx = 0
        for i in labels:
            i = tuple(i)
            i = int(i[0]), int(i[1])
            if i not in labels_id.keys():
                labels_id[i] = idx
                idx += 1
        return labels_id

    ## here I totally changed the __iter__ function to simplify the key mechanism, as I thought it was unnescarly commplex as there are only 2 classes no need for 12 keys

    def __iter__(self):
        self.iter_ = 0
        while self.iter_ < self.num_iters:
            classes = self.labels_set

            with open('output.txt', 'a') as file:
                file.write(f"labels_set {self.labels_set}\n")

            indices = []
            class_ = self.labels_set[self.counter]

            other_classes = np.delete(classes, self.counter,axis=0)

            #add class_ positives from a patients
            indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.num_pos_samples])
            self.used_label_indices_count[class_] += self.num_pos_samples

            #add negatives from patient's other condition
            label_ =  list(self.labels_dict.items())[0]
            label_ = label_[0]

            other_ = list(self.labels_dict.items())[1]
            other_ = other_[0]
            
            
            if self.num_neg_samples <= 1000:
                indices.extend(np.random.choice(self.label_to_indices[self.labels_dict[other_]], size=self.num_neg_samples, replace=False, p=None))
            
            else:
                print("self.num_neg_samples > 1000", self.num_neg_samples)
            
            if self.num_neg_others != 0:
                #remove patient's other condition from other_classes
                other_classes = other_classes[other_classes != self.labels_dict[other_]]
                np.random.shuffle(other_classes)
                other_classes = [self.labels_dict[key] for key, _ in self.labels_dict.items() if key[0] == label_[0] and key != label_]

                #Add  uniformly selected from other patients
                for other_class in other_classes:
                    if other_class != other_classes[-1]:
                        indices.extend(np.random.choice(self.label_to_indices[other_class], size = self.num_neg_others, replace=False, p=None))
                    else:
                        indices.extend(np.random.choice(self.label_to_indices[other_class],
                            size = self.batch_size - len(indices),
                            replace=False, p=None))

            self.used_label_indices_count[class_] += self.num_pos_samples

            if self.used_label_indices_count[class_] + self.num_pos_samples > len(self.label_to_indices[class_]):

                np.random.shuffle(self.label_to_indices[class_])

                self.used_label_indices_count[class_] = 0

            #advance class counter
            self.counter = (self.counter + 1) % len(self.labels_set)
            
            #Advance number of iterations counter; specific to data size of 20000
            self.iter_ += 1

            yield indices

    def __len__(self):
        return len(self.dataset) // self.num_pos_samples


def motion_dataloader(dataset, sampler, num_workers = 0, pin_memory = False):
    return DataLoader(
        dataset,
        batch_size=None,
        collate_fn=motion_collate,
        num_workers=0,
        pin_memory = pin_memory,
        sampler=sampler,
    )
    
def standard_motion_dataloader(dataset, batch_size: int, num_workers = 0, pin_memory = False):
    return DataLoader(
        dataset,
        batch_size=None,
        collate_fn=motion_collate,
        num_workers=num_workers,
        pin_memory = pin_memory,
        sampler=BatchSampler( RandomSampler(dataset),
                             batch_size=batch_size, drop_last=False),
        
    )
