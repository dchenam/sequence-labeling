import argparse
import numpy as np
import pandas as pd
from os import mkdir
from os.path import exists, join
from model import RNN_Model
from utils import load_timit, post_processing


def parse():
    parser = argparse.ArgumentParser(description="Timit Sequence Labeling")
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epoch', default=20, type=int, help='number of epochs')
    parser.add_argument('--layers', default="512,512", help='layer dimensions')
    parser.add_argument('--padding_size', default=500, type=int, help='padding size')
    parser.add_argument('--cell_type', default='lstm', choices=['rnn', 'lstm', 'gru'], help='rnn, lstm, or gru cells')
    parser.add_argument('--direction', default='bi', choices=['bi', 'uni'], help='uni or bidirectional')
    parser.add_argument('--data_type', default='mfcc', choices=['mfcc', 'fbank'], help='use fbank or mfcc')
    parser.add_argument('--data_dir', default='./timit', help='training data dir')
    parser.add_argument('--load', action='store_true', help='loads latest checkpoint')
    parser.add_argument('--output_dir', default='./models', help='directory of models')
    parser.add_argument('--testing', action='store_true', help='testing mode')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    config = parse()
    config.model_name = "{}-{}-{}-{}".format(config.data_type, config.direction, config.cell_type, str(config.layers))
    config.model_dir = join(config.output_dir, config.model_name)

    if not exists(config.output_dir):
        mkdir(config.output_dir)
    if not exists(config.model_dir):
        mkdir(config.model_dir)

    model = RNN_Model(config)

    if config.testing:
        test_ids, test_data, _ = load_timit(config, data_set='test')
        test_ids = np.expand_dims(test_ids, 1)

        predictions = model.test(test_data)
        predictions = np.expand_dims(post_processing(config, predictions, threshold=2), 1)

        outputs = np.append(test_ids, predictions, axis=1)
        df = pd.DataFrame(outputs).to_csv(join(config.model_dir, '{}.csv'.format(config.model_name)),
                                          index=False, header=['id', 'phone_sequence'])

    else:
        train_ids, train_data, train_labels = load_timit(config, data_set='train')
        model.train(train_data, train_labels)
