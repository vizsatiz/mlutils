import os
from multiprocessing.pool import Pool

from utils.io_utils import IOUtils
from utils.string_utils import StringUtils

data_dir = os.path.join("/Users", "vizsatiz", "Documents", "data", "messai-data-clean")
clean_dir = os.path.join("/Users", "vizsatiz", "Documents", "data", "messai-data-clean")
seq_dir = os.path.join("/Users", "vizsatiz", "Documents", "data", "messai-data-seq")
next_dir = os.path.join("/Users", "vizsatiz", "Documents", "data", "messai-data-next")
model_dir = os.path.join("/Users", "vizsatiz", "Documents", "data", "models")



from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy


def train():
    v, vs = create_vocabulary(create_word_list(), "{}/{}".format(model_dir, "vocab.pb"))
    s, n = create_sentence_corpus()
    X_train, y_train = create_vectors(v, vs, s, n)

    rnn_size = 128  # size of RNN
    seq_length = 4  # sequence length
    learning_rate = 0.005  # learning rate

    print('Build LSTM model.')
    from keras import regularizers
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, vs)))
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(rnn_size, activation="relu")))
    # model.add(Dropout(0.2))
    model.add(Dense(vs))
    model.add(Activation('softmax'))

    import datetime
    from keras.callbacks import TensorBoard
    optimizer = Adam(lr=learning_rate)
    # optimizer = RMSprop(lr=0.01)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [EarlyStopping(patience=10, monitor='val_loss'),
                 ModelCheckpoint(filepath=model_dir + "/" + 'my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5'
                                 , monitor='val_loss', verbose=0, mode='auto', period=2), tensorboard_callback]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")

    md = model
    md.summary()

    batch_size = 1000  # minibatch size
    num_epochs = 50  # number of epochs

    # fit the model
    md.fit(X_train, y_train,
           batch_size=batch_size,
           shuffle=True,
           epochs=num_epochs,
           callbacks=callbacks,
           validation_split=0.1)

    # save the model
    md.save(model_dir + "/" + 'my_model_generate_sentences.h5')





train()
# clean_data_to_remove_un_frequent_words(20)
# remove_duplicates()
# remove_multiple_ners()
