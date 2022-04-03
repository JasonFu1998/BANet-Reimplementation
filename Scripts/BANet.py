from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.backend import sum
from keras.callbacks import EarlyStopping

import tensorflow as tf
import warnings
from Preprocess import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(2)


# -----------------------------------------||||||||||||||||||||||||||||||||||||||||||||||||||

# Reference:

# "Learning Bodily and Temporal Attention in Protective Movement Behavior Detection"
#  arxiv preprint arxiv:1904.10824 (2019)

# "Automatic Detection of Protective Behavior in Chronic Pain Physical Rehabilitation: A Recurrent Neural Network Approach."
#  arXiv preprint arXiv:1902.08990 (2019).

# -----------------------------------------||||||||||||||||||||||||||||||||||||||||||||||||||


def crop(dimension, start, end):
    # Thanks to marc-moreaux on https://github.com/keras-team/keras/issues/890 who created this function:)
    # Crops (or slices) a Tensor on a given dimension from start to end
    # Example: to crop tensor x[:, :, 5: 10]
    # Call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func)


def build_model():
    
    body_num = 13  # number of body segments (different sensors) to consider

    # Model 1: Temporal information encoding model (keras Model API)
    single_input = Input(shape=(180, 2,))
    lstm_units = 8
    lstm1 = LSTM(lstm_units, return_sequences=True, implementation=1)(single_input)
    dropout1 = Dropout(0.5)(lstm1)
    lstm2 = LSTM(lstm_units, return_sequences=True, implementation=1)(dropout1)
    dropout2 = Dropout(0.5)(lstm2)
    lstm3 = LSTM(lstm_units, return_sequences=True, implementation=1)(dropout2)
    dropout3 = Dropout(0.5)(lstm3)
    TemporalProcessModel = Model(inputs=single_input, outputs=dropout3)
    # TemporalProcessModel.summary()

    # Model 2: Main structure, starting with independent temporal information encoding and attention learning
    # The input data is 180 time steps by 30 features (13 angles + 13 energies + 4 sEMGs)
    inputs = Input(shape=(180, 30,))
    # The information each body segment included is the angle and energy

    angle1 = crop(2, 0, 1)(inputs)
    acc1 = crop(2, 13, 14)(inputs)
    b1 = concatenate([angle1, acc1], axis=-1)
    angle_full_out1 = TemporalProcessModel(b1)
    # Temporal Attention Module for each body segment will starts with 1 X 1 Conv
    temporal_attention1 = Conv1D(1, 1, strides=1)(angle_full_out1)
    temporal_attention1 = Softmax(axis=-2, name='TemporalAttention1')(temporal_attention1)
    angel_attention_out1 = multiply([angle_full_out1, temporal_attention1])
    angel_attention_out1 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out1)
    blast1 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out1)

    angle2 = crop(2, 1, 2)(inputs)
    acc2 = crop(2, 14, 15)(inputs)
    b2 = concatenate([angle2, acc2], axis=-1)
    angle_full_out2 = TemporalProcessModel(b2)
    temporal_attention2 = Conv1D(1, 1, strides=1)(angle_full_out2)
    temporal_attention2 = Softmax(axis=-2, name='TemporalAttention2')(temporal_attention2)
    angel_attention_out2 = multiply([angle_full_out2, temporal_attention2])
    angel_attention_out2 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out2)
    blast2 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out2)

    angle3 = crop(2, 2, 3)(inputs)
    acc3 = crop(2, 15, 16)(inputs)
    b3 = concatenate([angle3, acc3], axis=-1)
    angle_full_out3 = TemporalProcessModel(b3)
    temporal_attention3 = Conv1D(1, 1, strides=1)(angle_full_out3)
    temporal_attention3 = Softmax(axis=-2, name='TemporalAttention3')(temporal_attention3)
    angel_attention_out3 = multiply([angle_full_out3, temporal_attention3])
    angel_attention_out3 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out3)
    blast3 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out3)

    angle4 = crop(2, 3, 4)(inputs)
    acc4 = crop(2, 16, 17)(inputs)
    b4 = concatenate([angle4, acc4], axis=-1)
    angle_full_out4 = TemporalProcessModel(b4)
    temporal_attention4 = Conv1D(1, 1, strides=1)(angle_full_out4)
    temporal_attention4 = Softmax(axis=-2, name='TemporalAttention4')(temporal_attention4)
    angel_attention_out4 = multiply([angle_full_out4, temporal_attention4])
    angel_attention_out4 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out4)
    blast4 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out4)

    angle5 = crop(2, 4, 5)(inputs)
    acc5 = crop(2, 17, 18)(inputs)
    b5 = concatenate([angle5, acc5], axis=-1)
    angle_full_out5 = TemporalProcessModel(b5)
    temporal_attention5 = Conv1D(1, 1, strides=1)(angle_full_out5)
    temporal_attention5 = Softmax(axis=-2, name='TemporalAttention5')(temporal_attention5)
    angel_attention_out5 = multiply([angle_full_out5, temporal_attention5])
    angel_attention_out5 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out5)
    blast5 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out5)

    angle6 = crop(2, 5, 6)(inputs)
    acc6 = crop(2, 18, 19)(inputs)
    b6 = concatenate([angle6, acc6], axis=-1)
    angle_full_out6 = TemporalProcessModel(b6)
    temporal_attention6 = Conv1D(1, 1, strides=1)(angle_full_out6)
    temporal_attention6 = Softmax(axis=-2, name='TemporalAttention6')(temporal_attention6)
    angel_attention_out6 = multiply([angle_full_out6, temporal_attention6])
    angel_attention_out6 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out6)
    blast6 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out6)

    angle7 = crop(2, 6, 7)(inputs)
    acc7 = crop(2, 19, 20)(inputs)
    b7 = concatenate([angle7, acc7], axis=-1)
    angle_full_out7 = TemporalProcessModel(b7)
    temporal_attention7 = Conv1D(1, 1, strides=1)(angle_full_out7)
    temporal_attention7 = Softmax(axis=-2, name='TemporalAttention7')(temporal_attention7)
    angel_attention_out7 = multiply([angle_full_out7, temporal_attention7])
    angel_attention_out7 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out7)
    blast7 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out7)

    angle8 = crop(2, 7, 8)(inputs)
    acc8 = crop(2, 20, 21)(inputs)
    b8 = concatenate([angle8, acc8], axis=-1)
    angle_full_out8 = TemporalProcessModel(b8)
    temporal_attention8 = Conv1D(1, 1, strides=1)(angle_full_out8)
    temporal_attention8 = Softmax(axis=-2, name='TemporalAttention8')(temporal_attention8)
    angel_attention_out8 = multiply([angle_full_out8, temporal_attention8])
    angel_attention_out8 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out8)
    blast8 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out8)

    angle9 = crop(2, 8, 9)(inputs)
    acc9 = crop(2, 21, 22)(inputs)
    b9 = concatenate([angle9, acc9], axis=-1)
    angle_full_out9 = TemporalProcessModel(b9)
    temporal_attention9 = Conv1D(1, 1, strides=1)(angle_full_out9)
    temporal_attention9 = Softmax(axis=-2, name='TemporalAttention9')(temporal_attention9)
    angel_attention_out9 = multiply([angle_full_out9, temporal_attention9])
    angel_attention_out9 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out9)
    blast9 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out9)

    angle10 = crop(2, 9, 10)(inputs)
    acc10 = crop(2, 22, 23)(inputs)
    b10 = concatenate([angle10, acc10], axis=-1)
    angle_full_out10 = TemporalProcessModel(b10)
    temporal_attention10 = Conv1D(1, 1, strides=1)(angle_full_out10)
    temporal_attention10 = Softmax(axis=-2, name='TemporalAttention10')(temporal_attention10)
    angel_attention_out10 = multiply([angle_full_out10, temporal_attention10])
    angel_attention_out10 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out10)
    blast10 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out10)

    angle11 = crop(2, 10, 11)(inputs)
    acc11 = crop(2, 23, 24)(inputs)
    b11 = concatenate([angle11, acc11], axis=-1)
    angle_full_out11 = TemporalProcessModel(b11)
    temporal_attention11 = Conv1D(1, 1, strides=1)(angle_full_out11)
    temporal_attention11 = Softmax(axis=-2, name='TemporalAttention11')(temporal_attention11)
    angel_attention_out11 = multiply([angle_full_out11, temporal_attention11])
    angel_attention_out11 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out11)
    blast11 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out11)

    angle12 = crop(2, 11, 12)(inputs)
    acc12 = crop(2, 24, 25)(inputs)
    b12 = concatenate([angle12, acc12], axis=-1)
    angle_full_out12 = TemporalProcessModel(b12)
    temporal_attention12 = Conv1D(1, 1, strides=1)(angle_full_out12)
    temporal_attention12 = Softmax(axis=-2, name='TemporalAttention12')(temporal_attention12)
    angel_attention_out12 = multiply([angle_full_out12, temporal_attention12])
    angel_attention_out12 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out12)
    blast12 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out12)

    angle13 = crop(2, 12, 13)(inputs)
    acc13 = crop(2, 25, 26)(inputs)
    b13 = concatenate([angle13, acc13], axis=-1)
    angle_full_out13 = TemporalProcessModel(b13)
    temporal_attention13 = Conv1D(1, 1, strides=1)(angle_full_out13)
    temporal_attention13 = Softmax(axis=-2, name='TemporalAttention13')(temporal_attention13)
    angel_attention_out13 = multiply([angle_full_out13, temporal_attention13])
    angel_attention_out13 = Lambda(lambda x: sum(x, axis=1, keepdims=True))(angel_attention_out13)
    blast13 = Permute((2, 1), input_shape=(1, lstm_units))(angel_attention_out13)

    # Feature concat for Bodily Attention Learning
    # The size of the output from each body segment is k X 1, while k is the number of LSTM hidden units
    # During prior experiments, we found that it is better to keep the dimension k instead of merging them into one

    data = concatenate([blast1, blast2, blast3, blast4, blast5, blast6, blast7, blast8,
                        blast9, blast10, blast11, blast12, blast13
                        ], axis=2)
    # Handy and sufficient Bodily Attention Module
    a = Dense(body_num, activation='tanh')(data)
    a = Dense(body_num, activation='softmax', name='bodyAttention')(a)
    attention_result = multiply([data, a])
    attention_result = Flatten()(attention_result)
    output = Dense(2, activation='softmax')(attention_result)

    model = Model(inputs=inputs, outputs=output)
    # model.summary()

    return model

    # Main Implementation Part


if __name__ == '__main__':

    results = []
    for person in range(1, 24):

        x_train, x_valid, y_train, y_valid = load_data(person)

        # callback 1: Save the better result after each epoch,
        checkPointer = keras.callbacks.ModelCheckpoint(filepath='Subject' + str(person) + '.hdf5',
                                                       monitor='val_binary_accuracy', verbose=1,
                                                       save_best_only=True)

        # callback 2: Stop if Acc = 1
        class EarlyStoppingByValAcc(keras.callbacks.Callback):
            def __init__(self, monitor='val_acc', value=1.00000, verbose=0):
                super(keras.callbacks.Callback, self).__init__()
                self.monitor = monitor
                self.value = value
                self.verbose = verbose

            def on_epoch_end(self, epoch, logs={}):
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
                if current == self.value:
                    if self.verbose > 0:
                        print("Epoch %05d: early stopping THR" % epoch)
                        self.model.stop_training = True


        callbacks = [
            EarlyStoppingByValAcc(monitor='val_binary_accuracy', value=1.00000, verbose=1),
            checkPointer
        ]

        model = build_model()
        ada = tf.keras.optimizers.Adam(learning_rate=0.003)
        model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=ada,
                      metrics=['binary_accuracy'])

        H = model.fit(x_train, y_train,
                      batch_size=40,
                      epochs=1,
                      shuffle=False,
                      callbacks=callbacks,
                      validation_data=(x_valid, y_valid))

        print('---This is result for %s th subject---' % person)
        model.load_weights('Subject' + str(person) + '.hdf5')
        y_pred = np.argmax(model.predict(x_valid, batch_size=15), axis=1)
        y_true = np.argmax(y_valid, axis=1)
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
        class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None) * 100) * 0.01
        print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f1)))
        results.append(cf_matrix)

    with open('results.txt', 'w') as f:
        for i in results:
            f.write(str(i))
