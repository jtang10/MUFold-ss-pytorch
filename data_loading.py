from __future__ import print_function, division
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class Protein_Dataset(Dataset):
    def __init__(self, relative_path, datalist_addr, max_seq_len=300, padding=True, feature_size=66):
        self.relative_path = relative_path
        self.protein_list = self.read_list(relative_path + datalist_addr)
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.feature_size = feature_size
        self.dict_ss = {key: value for (key, value) in \
            zip(['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], range(8))}

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        protein_name = self.protein_list[idx]
        features, labels, length = self.read_protein(protein_name, self.relative_path, self.max_seq_len, self.padding)
        return torch.from_numpy(features).float(), torch.from_numpy(labels).long(), length

    def read_list(self, filename):
        """Given the filename storing all protein names, return a list of protein names.
        """
        with open(filename) as f:
            proteins_names = f.read().splitlines()
        return proteins_names

    def read_protein(self, protein_name, relative_path, max_seq_len=300, padding=False):
        """Given a protein name, return the ndarray of features [1 x seq_len x n_features]
        and labels [1 x seq_len].
        """
        features_addr = relative_path + '66FEAT/' + protein_name + '.66feat'
        labels_addr = relative_path + 'Angles/' + protein_name + '.ang'

        protein_features = np.loadtxt(features_addr)
        protein_labels = []
        with open(labels_addr) as f:
            next(f)
            for i, line in enumerate(f):
                line = line.split('\t')
                if line[0] == '0':
                    # 0 means the current ss label exists.
                    protein_labels.append(self.dict_ss[line[3]])
        protein_labels = np.array(protein_labels)
        # print(protein_labels.shape)
        protein_length = protein_labels.shape[0]
        if padding:
            # if features passes max_seq_len, cutoff
            if protein_features.shape[0] >= max_seq_len:
                protein_features = protein_features[:max_seq_len, :]
                protein_labels = protein_labels[:max_seq_len]
                protein_length = max_seq_len
            # else, zero-pad to max_seq_len
            else:
                padding_length = max_seq_len - protein_features.shape[0]
                protein_features = np.pad(protein_features, ((0, padding_length), (0, 0)),
                                          'constant', constant_values=((0, 0), (0, 0)))
                protein_labels = np.pad(protein_labels, (0, padding_length), 'constant', constant_values=(0, 0))

        return protein_features, protein_labels, protein_length

if __name__ == '__main__':
    from operator import mul
    from functools import reduce
    SetOf7604Proteins_path = '../data/SetOf7604Proteins/'
    trainList_addr = 'trainList'
    validList_addr = 'validList'
    testList_addr = 'testList'

    protein_dataset = Protein_Dataset(SetOf7604Proteins_path, testList_addr, max_seq_len=698, padding=False)
    dataloader = DataLoader(protein_dataset, batch_size=1, shuffle=False, num_workers=4)

    start = time.time()
    size1 = 0
    size2 = 0
    for i, sample_batched in enumerate(dataloader):
        features, labels, lengths = sample_batched
        max_length = max(lengths.numpy())
        features = Variable(features[:, :, :max_length])
        print(type(features.data))
        labels = Variable(labels[:, :max_length])
        size2 += reduce(mul, features.size())
        break
    print("Spent {:.3f}s".format(time.time() - start))