from __future__ import print_function
import numpy as np
import time

def read_list(filename):
    """Read the protein list file and return a list of protein name"""
    with open(filename) as f:
        proteins_names = f.read().splitlines()
    return proteins_names


def get_proteins_length(proteins_list, relative_path):
    """Given a list of proteins, return the lengths for each protein."""
    proteins_length = []
    for protein_name in proteins_list:
        protein_addr = relative_path + '66FEAT/' + protein_name + '.66feat'
        with open(protein_addr) as f:
            for i, l in enumerate(f):
                pass
            protein_length = i + 1
            proteins_length.append(protein_length)

    return np.array(proteins_length)


def read_protein(protein_name, relative_path, max_seq_len=300, padding=False):
    """Given a protein name, return the ndarray of features [1 x seq_len x n_features]
    and labels [1 x seq_len].
    """
    ss = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    dict_ss = {key: value for (key, value) in zip(ss, range(8))}
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
                protein_labels.append(dict_ss[line[3]])
    protein_labels = np.array(protein_labels).transpose()
    if padding:
        # if features passes max_seq_len, cutoff
        if protein_features.shape[0] >= max_seq_len:
            protein_features = protein_features[:max_seq_len, :]
            protein_labels = protein_labels[:max_seq_len]
        # else, zero-pad to max_seq_len
        else:
            padding_length = max_seq_len - protein_features.shape[0]
            protein_features = np.pad(protein_features, ((0, padding_length), (0, 0)),
                                      'constant', constant_values=((0, 0), (0, 0)))
            protein_labels = np.pad(protein_labels, (0, padding_length), 'constant', constant_values=(0, 0))

    protein_features = np.expand_dims(protein_features, axis=0)
    protein_labels = np.expand_dims(protein_labels, axis=0)

    return protein_features, protein_labels


def minibatches(protein_list, relative_path, batch_size, max_seq_len=None, shuffle=True):
    """Given a list of proteins' name and length, return a generateor of batch 
    of protein features and labels, zero-padded to the longest sequence within 
    the batch.
    """
    data_size = len(protein_list)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_start in np.arange(0, data_size, batch_size):
        batch_indices = indices[batch_start : (batch_start + batch_size)]
        yield get_batch(protein_list, relative_path, batch_indices, max_seq_len)


def get_batch(protein_list, relative_path, batch_indices, max_seq_len=None, n_features=66):
    """Given indices of the desired protein, return the batched and zero-padded
    features [batch_size x max_seq_len x n_features] and 
    labels [batch_size x max_seq_len].
    If max_seq_len is given, zero-pad to max_seq_len, otherwise zero-pad to the
    longest sequence within the batches.
    """
    proteins_length = get_proteins_length(protein_list, relative_path)
    batch_names = np.array(protein_list)[batch_indices]
    batch_size = len(batch_indices)
    if not max_seq_len:
        max_seq_len = max(proteins_length[batch_indices])
    protein_features = np.zeros((batch_size, max_seq_len, n_features))
    protein_labels = np.zeros((batch_size, max_seq_len))
    for i, protein_name in enumerate(batch_names):
        protein_features[i, :, :], protein_labels[i, :] = read_protein(protein_name, relative_path, max_seq_len, padding=True)

    return protein_features, protein_labels


def read_protein_old(prot_name, relative_path, expand_dims=False, CASP=False):
    """Given a protein name, return a matrix of its features, corresponding
       second structure information, sequence length and mask.
    This is a sub-function of reading entire list of data and can also be
    used to read individual protein.
    """
    ss = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    dict_ss = {key: value for (key, value) in zip(ss, range(8))}
    features_addr = relative_path + '66FEAT/' + prot_name + '.66feat'
    labels_addr = relative_path + 'Angles/' + prot_name + '.ang'
    protein_features = np.loadtxt(features_addr)
    protein_labels = []
    delete_idx = []

    with open(labels_addr) as f:
        next(f)
        for i, line in enumerate(f):
            line = line.split('\t')
            if line[0] == '0':
                # 0 means the current ss label exists.
                protein_labels.append(dict_ss[line[3]])
            else:
                # 1 means current residue has no ss label and according features will be removed.
                delete_idx.append(i)

    if CASP:
        protein_features = np.delete(protein_features, delete_idx, axis=0)
    protein_labels = np.array(protein_labels).transpose()
    seq_len = protein_labels.shape[0]
    mask = np.ones((seq_len,), dtype=np.float32)

    if expand_dims:
        protein_features = np.expand_dims(protein_features, axis=0)
        protein_labels = np.expand_dims(protein_labels, axis=0)
        seq_len = np.array([seq_len])
        mask = np.expand_dims(mask, axis=0)

    return protein_features, protein_labels, seq_len, mask


def generate_batch(prot_list, relative_path, max_seq_length=300, batch_size=64):
    """Given the list of protein name and length, generate a batch of feature 
       matrix and corresponding secondary structure label. Zero pad or cutoff 
       feature matrix based on max_seq_length.
    """
    while True:
        num_list = len(prot_list)
        batch_idx = np.random.randint(0, num_list, batch_size)

        # Create the batch matrix with all zeros for zero-padding.
        proteins_batch = np.zeros((batch_size, max_seq_length, 66),
                                  dtype=np.float32)
        ss_labels_batch = np.zeros((batch_size, max_seq_length),
                                   dtype=np.int32)
        # mask_batch = np.zeros((batch_size, max_seq_length), dtype=np.float32)
        # batch_seq_len = []

        for i, j in enumerate(batch_idx):
            protein_name = prot_list[j]
            protein_features, ss_labels, seq_len, mask = read_protein_old(protein_name, relative_path)
            min_idx = min(max_seq_length, seq_len)
            proteins_batch[i, :min_idx, :] = protein_features[:min_idx, :]
            ss_labels_batch[i, :min_idx] = ss_labels[:min_idx]
            # mask_batch[i, :min_idx] = mask[:min_idx]
            # batch_seq_len.append(min_idx)

        # batch_seq_len = np.asarray(batch_seq_len, dtype=np.int32)
        yield proteins_batch, ss_labels_batch#, batch_seq_len, mask_batch


def read_data(prot_list, relative_path, max_seq_length=300):
    """Given lists of training, validation and test datasets, return the 
       ndarrays of all data, labels and lengths.
    """
    num_list = len(prot_list)
    proteins_all = np.zeros((num_list, max_seq_length, 66), dtype=np.float32)
    ss_labels_all = np.zeros((num_list, max_seq_length), dtype=np.int32)
    mask_all = np.zeros((num_list, max_seq_length), dtype=np.float32)
    seq_lens_all = []

    for i, protein_name in enumerate(prot_list):
        protein_features, ss_labels, seq_len, mask = read_protein_old(protein_name, relative_path)
        min_idx = min(max_seq_length, seq_len)
        proteins_all[i, :min_idx, :] = protein_features[:min_idx, :]
        ss_labels_all[i, :min_idx] = ss_labels[:min_idx]
        mask_all[i, :min_idx] = mask[:min_idx]
        seq_lens_all.append(min_idx)
    seq_lens_all = np.asarray(seq_lens_all, dtype=np.int32)

    return proteins_all, ss_labels_all, seq_lens_all, mask_all

##################
# Test Functions #
##################
def test_read_protein(protein_list, relative_path):
    test_protein = protein_list[0]
    features, labels = read_protein(test_protein, relative_path, padding=False)
    print("Protein Name:", test_protein)
    print("features:", features.shape)
    print("labels:", labels.shape)


def test_get_batch(protein_list, relative_path):
    start = time.time()
    for i in range(19):
        num_list = len(protein_list)
        batch_indices = np.random.randint(0, num_list, 64)
        pf, pl = get_batch(protein_list, relative_path, batch_indices, max_seq_len=400)
        print(pf.shape, pl.shape)
    print("Spent {:.2f}s".format(time.time() - start))


def test_generate_batch(protein_list, relative_path, max_seq_len):
    print("Old method test started")
    start = time.time()
    for i, batch in enumerate(generate_batch(protein_list, relative_path, max_seq_len)):
        features, labels = batch
        print(features.shape, labels.shape)
        if i == 19:
            break
    print("Old method finished {} batches in {:.2f}s".format(i, time.time() - start))

def test_minibatches(protein_list, relative_path, max_seq_len):
    print("New method test started")
    start = time.time()
    for i, batch in enumerate(minibatches(protein_list, relative_path, 64, max_seq_len)):
        features, labels = batch
        print(features.shape, labels.shape)
    print("New method finished {} batches in {:.2f}s".format(i, time.time() - start))

if __name__ == '__main__':
    SetOf7604Proteins_path = '../data/SetOf7604Proteins/'
    trainList_addr = SetOf7604Proteins_path + 'trainList'
    validList_addr = SetOf7604Proteins_path + 'validList'
    testList_addr = SetOf7604Proteins_path + 'testList'

    casp11_path = '../data/CASP11/'
    casp11List_addr = casp11_path + 'proteinList'
    casp12_path = '../data/CASP12/'
    casp12List_addr = casp12_path + 'proteinList'

    trainList = read_list(trainList_addr)
    validList = read_list(validList_addr)
    testList = read_list(testList_addr)
    casp12List = read_list(casp12List_addr)

    # print(get_proteins_length(validList, SetOf7604Proteins_path))
    # test_read_protein(validList, SetOf7604Proteins_path)
    test_get_batch(validList, SetOf7604Proteins_path)
    # test_minibatches(validList, SetOf7604Proteins_path, 400)
    test_generate_batch(validList, SetOf7604Proteins_path, 400)
