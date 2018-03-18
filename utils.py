import numpy as np
import pickle
from itertools import groupby
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from os.path import join


# 'L' maps to 'sil
def trim_sil(s):
    if s.endswith("L"): s = s[:-1]
    if s.startswith("L"): s = s[1:]
    return s


# convert to list of sequences by id
def load_sequences(input_data):
    sequences = {}
    for i, value in enumerate(input_data):
        line = value[0].split('_')
        seq_name = str(line[0] + '_' + line[1])
        if not seq_name in sequences:
            sequences[seq_name] = []
        sequences[seq_name].append(value[1])
    return sequences


# convert to list of labels by id, also maps 48 to 39 phonemes
def load_labels(input_data, label_mapping, mapping_39):
    labels = {}
    for i, value in enumerate(input_data):
        line = value[0].split('_')
        seq_name = str(line[0] + '_' + line[1])
        if not seq_name in labels:
            labels[seq_name] = []
        labels[seq_name].append(mapping_39[label_mapping[value[0]]])
    return labels


def load_timit(config, data_set='train'):
    print('loading...')

    path = config.data_dir
    kind = config.data_type
    padding_size = config.padding_size

    mapping_39, label_mapping = {}, {}
    input_data = []

    if kind == 'mfcc':
        data_path = join(path, '{}/{}.ark'.format(kind, data_set))
    else:
        data_path = join(path, '{}/{}.ark'.format(kind, data_set))

    with open(join(path, 'label/train.lab')) as labels:
        for line in labels:
            line = line.strip('\n').split(',')
            label_mapping[line[0]] = line[1]

    with open(join(path, 'phones/48_39.map')) as m:
        for line in m:
            line = line.strip('\n').split('\t')
            mapping_39[line[0]] = line[1]

    with open(data_path) as d:
        for line in d:
            line = line.strip('\n').split(' ')
            input_data.append([line[0], [float(x) for x in line[1:]]])

    data, labels, ids = [], [], []

    if data_set == 'train':
        id_data = load_sequences(input_data)
        ids = list(id_data.keys())
        data = list(id_data.values())
        ids_labels = load_labels(input_data, label_mapping, mapping_39)
        if ids != list(ids_labels.keys()):
            raise ValueError('data ids do not match label ids')
        labels = list(ids_labels.values())

        # Pre-Process Labels (Encode, Pad, One-Hot)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(mapping_39.values()))
        labels = [tokenizer.texts_to_sequences(label) for label in labels]
        labels = pad_sequences(labels, maxlen=padding_size, padding='pre', dtype='int32')
        labels = np.array([tokenizer.sequences_to_matrix(label) for label in labels])
        print('label shape: ', labels.shape)

        phone_to_idx = tokenizer.word_index
        idx_to_phone = {v: k for k, v in phone_to_idx.items()}

        with open(join(config.model_dir, 'phone_to_idx.pkl'), 'wb') as f:
            pickle.dump(phone_to_idx, f)

        with open(join(config.model_dir, 'idx_to_phone.pkl'), 'wb') as f:
            pickle.dump(idx_to_phone, f)
    else:
        id_data = load_sequences(input_data)
        ids = list(id_data.keys())
        data = list(id_data.values())

    print('pre-processing...')

    # Pre-Process Sequences (Standardize, Pad)
    scaled_data = []
    for sample in data:
        scaler = StandardScaler()
        scaled_data.append(scaler.fit_transform(sample))

    scaled_data = pad_sequences(scaled_data, maxlen=padding_size, padding='pre', dtype='float64')
    print('data shape: ', scaled_data.shape)

    return np.array(ids), np.array(scaled_data), np.array(labels)


# Post-Process (39 Phonemes to Alphabet, Remove Consecutive Duplicates w/ Threshold, Trim Sil)
def post_processing(config, predictions, threshold=1):
    phoneme_to_char = {}
    with open(join(config.data_dir, 'phones/48phone_char.map')) as m:
        for line in m:
            line = line.strip('\n').split('\t')
            phoneme_to_char[line[0]] = line[2]

    idx_to_phone = pickle.load(open(join(config.model_dir, 'idx_to_phone.pkl'), 'rb'))

    output = []
    for item in predictions:
        x = list(map(lambda x: phoneme_to_char[idx_to_phone[x]], item))
        x = ''.join([j for j, k in groupby(x) if sum(1 for i in k) > threshold])
        x = trim_sil(x)
        output.append(x)

    return np.array(output)
