from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, InputLayer, Masking
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.callbacks import TensorBoard, EarlyStopping
from os.path import join


class RNN_Model():
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.layers = [int(l) for l in config.layers.split(',')]
        self.direction = config.direction
        self.cell_type = config.cell_type
        self.num_classes = 40
        self.model_dir = config.model_dir
        self.model_name = config.model_name

        if config.data_type == 'mfcc':
            self.input_dim = (config.padding_size, 39)
        else:
            self.input_dim = (config.padding_size, 69)

        self.model = self.build()

    def rnn_layer(self, dim):
        if self.cell_type == 'rnn':
            return SimpleRNN(dim, dropout=0.25, return_sequences=True, activation='relu')
        elif self.cell_type == 'lstm':
            return LSTM(dim, dropout=0.25, return_sequences=True, activation='tanh', implementation=2)
        elif self.cell_type == 'gru':
            return GRU(dim, dropout=0.25, return_sequences=True, activation='relu', implementation=2)

    def build(self):
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_dim))
        model.add(Masking())
        for layer in self.layers:
            if self.direction == 'uni':
                model.add(self.rnn_layer(layer))
            elif self.direction == 'bi':
                model.add(Bidirectional(self.rnn_layer(int(layer / 2))))
        model.add(TimeDistributed(Dense(self.num_classes, activation='softmax')))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        model.summary()
        return model

    def train(self, x, y):
        tensorboard = TensorBoard(log_dir=self.model_dir + '/logs', batch_size=self.batch_size, write_graph=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        self.model.fit(x, y, self.batch_size, self.epoch, shuffle=True, callbacks=[tensorboard, early_stopping],
                       validation_split=0.2)

        self.model.save_weights(
            join(self.model_dir, self.model_name + '.h5'))

    def test(self, x):
        self.model.load_weights(
            join(self.model_dir, self.model_name + '.h5'))
        predictions = self.model.predict_classes(x)
        return predictions
