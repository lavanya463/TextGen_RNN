import glob
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import re

from keras.utils import plot_model

ENTIRE_DATA_FILE = "./data/entire_series.txt"
# uncomment to work with one txt file
# ENTIRE_DATA_FILE = "./data/1.txt"


def remove_special_chars():
    string = open(ENTIRE_DATA_FILE).read()
    new_str = re.sub('[^a-zA-Z0-9\n.?!, ]', ' ', string)
    open(ENTIRE_DATA_FILE, 'w').write(new_str)


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
    return ''.join(y_char)


def create_lstm_model(vocab_size):
    model = Sequential()
    model.add(LSTM(256, input_shape=(None, vocab_size), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    return model


def create_gru_model(vocab_size):
    model = Sequential()
    model.add(GRU(128,input_shape=(None, vocab_size), return_sequences=True))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    return model


def create_simplernn_model(vocab_size):
    model = Sequential()
    model.add(SimpleRNN(128,input_shape=(None, vocab_size), return_sequences=True))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    return model


def train_model(model, x, y, size_of_batch):

    """
    file_path = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(x, y, epochs=1000, batch_size=size_of_batch,callbacks=callbacks_list, verbose=1)
    """
    model.fit(x, y, epochs=1, batch_size=size_of_batch, verbose=1)
    # model.save_weights('checkpoint_layer_hidden_epoch_{}.hdf5'.format(100))

    return model


def main():

    # For initial run club the 5 books into a single book
    # combine_books()
    # pre-processing the data
    # remove_empty_lines()
    # remove_special_chars()

    window_size = 150
    x, y, vocab_size, ix_to_char = load_data(ENTIRE_DATA_FILE, window_size)

    # create simple RNN model
    # model = create_simplernn_model(vocab_size)

    # create the LSTM model
    # model = create_lstm_model(vocab_size)

    # create GRU model
    model = create_gru_model(vocab_size)

    # define the optimizer
    #optim = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # optim = 'adam'
    optim = 'adagrad'

    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=optim)

    # plot the RNN model
    model.summary()
    plot_model(model, to_file="RNN_model", show_shapes=True)

    print("first generation")
    text = generate_text(model, 100, vocab_size, ix_to_char)
    print(text)

    # Generate some sample before training to know how bad it is!
    # generate_text(model, 30, vocab_size, ix_to_char)

    model = train_model(model, x, y, 1000)

    # print("loading pre trained weights")
    # if you have the weigths then load the weigths, no training required
    # model.load_weights("weights-improvement-999-0.8728.hdf5")

    # generate the text, prediction stage
    print("final generation")
    text = generate_text(model, 500, vocab_size, ix_to_char)
    print(text)


if __name__ == "__main__":
    main()
