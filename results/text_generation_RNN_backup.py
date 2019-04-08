import glob
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
import argparse

from tensorflow.python.estimator import keras

ENTIRE_DATA_FILE = "./data/entire_series.txt"


def combine_books():
    read_files = glob.glob("./data/*.txt")
    with open(ENTIRE_DATA_FILE, "wb") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())


def remove_empty_lines():
    with open(ENTIRE_DATA_FILE) as file_handle:
        lines = file_handle.readlines()

    with open(ENTIRE_DATA_FILE, 'w') as file_handle:
        lines = filter(lambda x: x.strip(), lines)
        file_handle.writelines(lines)


def load_data(data_dir, seq_length):
    # loads the data and returns input sequence, target sequence, vocabulary size and index of character array
    data = open(data_dir, 'r').read()
    chars = list(set(data))
    vocab_size = len(chars)

    # print(data[:2000])

    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(vocab_size))

    ix_to_char = {ix: char for ix, char in enumerate(chars)}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}

    x = np.zeros((len(data)//seq_length, seq_length, vocab_size))
    y = np.zeros((len(data)//seq_length, seq_length, vocab_size))
    for i in range(0, len(data)//seq_length):
        x_sequence = data[i*seq_length:(i+1)*seq_length]
        x_sequence_ix = [char_to_ix[value] for value in x_sequence]
        input_sequence = np.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            input_sequence[j][x_sequence_ix[j]] = 1.
            x[i] = input_sequence

        y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
    return x, y, vocab_size, ix_to_char


def generate_text(model, length, vocab_size, ix_to_char):
    # starting with random character
    ix = [np.random.randint(vocab_size)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, vocab_size))
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][ix[-1]] = 1
        # print(ix_to_char[ix[-1]])
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


def create_model(vocab_size):
    model = Sequential()
    model = Sequential()
    model.add(LSTM(300, input_shape=(None, vocab_size), return_sequences=True))
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    optimizer_def = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer_def)

    return model


def train_model(model,x, y, size_of_batch, gen_length, vocab_size, ix_to_char):
    nb_epoch = 0

    '''
    while True:
        print('\n\nEpoch: {}\n'.format(nb_epoch))
        model.fit(x, y, batch_size=size_of_batch, verbose=1, nb_epoch=1)
        nb_epoch += 1
        generate_text(model, gen_length, vocab_size, ix_to_char)
        if nb_epoch % 10 == 0:
            model.save_weights('checkpoint_layer_hidden_epoch_{}.hdf5'.format(nb_epoch))
    '''
    model.fit(x, y, epochs=100, batch_size=size_of_batch, verbose=1)
    model.save_weights('checkpoint_layer_hidden_epoch_{}.hdf5'.format(100))

    return model


def main():

    # For initial run club the 5 books into a single book
    # combine_books()
    # remove_empty_lines()

    x, y, vocab_size, ix_to_char = load_data(ENTIRE_DATA_FILE, 50)

    # print(ix_to_char)
    # create the LSTM model
    model = create_model(vocab_size)

    # Generate some sample before training to know how bad it is!
    # generate_text(model, 30, vocab_size, ix_to_char)

    model = train_model(model, x, y, 1000, 30, vocab_size, ix_to_char)

    # print("loading pre trained weights")
    # if you have the weigths then load the weigths, no training required
    # model.load_weights("checkpoint_layer_hidden_epoch_1000.hdf5")

    print("final generation")
    text = generate_text(model, 1000, vocab_size, ix_to_char)
    print(text)


if __name__ == "__main__":
    main()
