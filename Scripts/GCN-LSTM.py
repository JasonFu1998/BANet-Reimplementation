# -------------------------------------------------------------------
# GitHub: https://github.com/Mvrjustid/IMWUT-Hierarchical-HAR-PBD
#
# @article{wang2021leveraging,
#     title={Leveraging Activity Recognition to Enable Protective Behavior Detection in Continuous Data},
#     author={Wang, Chongyang and Gao, Yuan and Mathur, Akhil and Williams, Amanda C. DE C. and Lane, Nicholas D and Bianchi-Berthouze, Nadia},
#     journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)},
#     volume={5},
#     number={2},
#     DOI={10.1145/3463508},
#     publisher={ACM},
#     year={2021}}
#
# -------------------------------------------------------------------

import warnings
import tensorflow as tf
from sklearn.metrics import *
from keras.layers import *
from keras.models import *
from keras.callbacks import EarlyStopping
from scipy.linalg import fractional_matrix_power
from test import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def output_of_adj_mul(input_shape):
    return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])


def adj_mul(x):
    adj_norm = make_graph()
    x = tf.cast(x, tf.float64)  # This step could be removed in earlier Tensorflow versions.
    return tf.matmul(adj_norm, x)


def build_model(time_step, body_num, feature_dim, gcn_units, lstm_units, class_num):
    # time_step is the length of current input data segment
    # body_num is the number of nodes/joints of the input graph
    # feature_num is the feature dimension of each node/joint
    # gcn/lstm_units is the number of gcn and lstm layers' units
    # class_num is the number of categories

    # LSTM with three layers.
    SingleInput = Input(shape=(time_step, body_num * gcn_units))
    LSTM1 = LSTM(lstm_units, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False,
                 use_bias=True, return_sequences=True)(SingleInput)
    Dropout1 = Dropout(0.5)(LSTM1)
    LSTM2 = LSTM(lstm_units, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False,
                 use_bias=True, return_sequences=True)(Dropout1)
    Dropout2 = Dropout(0.5)(LSTM2)
    LSTM3 = LSTM(lstm_units, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False,
                 use_bias=True, return_sequences=False)(Dropout2)
    Dropout3 = Dropout(0.5)(LSTM3)
    TemporalProcessModel = Model(inputs=[SingleInput], outputs=[Dropout3])

    # GCN with three layers
    Inputs = Input(shape=(time_step, body_num, feature_dim,), name='main_inputs')
    Dense1 = TimeDistributed(Conv1D(gcn_units, 1, activation='relu'))(Inputs)
    Dense1 = Dropout(0.5)(Dense1)
    Dense2 = Lambda(adj_mul, output_shape=output_of_adj_mul)(Dense1)
    Dense2 = TimeDistributed(Conv1D(gcn_units, 1, activation='relu'))(Dense2)
    Dense2 = Dropout(0.5)(Dense2)
    Dense3 = Lambda(adj_mul, output_shape=output_of_adj_mul)(Dense2)
    Dense3 = TimeDistributed(Conv1D(gcn_units, 1, activation='relu'))(Dense3)
    Dense3 = Dropout(0.5)(Dense3)
    GcnOutput = Reshape((time_step, body_num * gcn_units), )(Dense3)
    TemporalOutput = TemporalProcessModel(GcnOutput)
    TemporalOutput = Dense(class_num, activation='softmax')(TemporalOutput)
    model = Model(inputs=[Inputs], outputs=[TemporalOutput])

    model.summary()
    return model


def make_graph():
    # This function is used for the situation where the target graph can be easily and manually defined.
    # Here is an example how we define the graph of human skeleton with 14 nodes.
    # 1. define the 14 x 14 adjacency matrix
    adj = np.zeros((14, 14))
    adj[0, 1] = 1
    adj[1, 0] = 1
    adj[1, 2] = 1
    adj[2, 1] = 1
    adj[1, 3] = 1
    adj[3, 1] = 1
    adj[3, 5] = 1
    adj[5, 3] = 1
    adj[5, 7] = 1
    adj[7, 5] = 1
    adj[2, 4] = 1
    adj[4, 2] = 1
    adj[4, 6] = 1
    adj[6, 4] = 1
    adj[1, 9] = 1
    adj[9, 1] = 1
    adj[1, 8] = 1
    adj[8, 1] = 1
    adj[9, 11] = 1
    adj[11, 9] = 1
    adj[11, 13] = 1
    adj[13, 11] = 1
    adj[8, 10] = 1
    adj[10, 8] = 1
    adj[10, 12] = 1
    adj[12, 10] = 1

    # 2. define the diagonal 14 x 14 degree matrix
    degree = np.zeros((14, 14))
    degree[0, 0] = 1
    degree[1, 1] = 5
    degree[2, 2] = 2
    degree[3, 3] = 2
    degree[4, 4] = 2
    degree[5, 5] = 2
    degree[6, 6] = 1
    degree[7, 7] = 1
    degree[8, 8] = 2
    degree[9, 9] = 2
    degree[10, 10] = 2
    degree[11, 11] = 2
    degree[12, 12] = 1
    degree[13, 13] = 1

    # 3. compute and output the normalized adjacency matrix, referring to Equation 1 in the paper.
    adj_i = np.identity(14)
    degree_power = fractional_matrix_power(degree + adj_i, -0.5)
    adj_norm = np.matmul(np.matmul(degree_power, adj + adj_i), degree_power)

    return adj_norm


def build_callbacks(person):
    # model_name is a string indicating the name of current training period.

    # callback 1: Save the model with improved results after each epoch,
    check_pointer = keras.callbacks.ModelCheckpoint(filepath='Dancer' + str(person) + '.hdf5',
                                                    monitor='val_binary_accuracy', verbose=1, save_best_only=True)

    # callback 2: Stop if Acc=1
    class EarlyStoppingByValAcc(keras.callbacks.Callback):
        def __init__(self, monitor, value, verbose):
            super(keras.callbacks.Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            if current == self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                    self.model.stop_training = True

    callbacks = [EarlyStoppingByValAcc(monitor='val_binary_accuracy', value=1.0000, verbose=1), check_pointer
                 # TensorBoard(log_dir='dir/logfolder') # Add yours to check the TensorBoard
                 ]
    return callbacks


def combine(adj, x, time_step, node_num, feature_num):
    # Transfer the input matrix X (N x T x D) into graph sequences (N x T x node_num x feature_num) as input
    # assume X=[X_coordinate, Y_coordinate, Z_coordinate].
    # Adj is Adjacency matrix of (None, node_num, node_num).
    # time_step is the length of each data segment.
    # node_num is the number of nodes/joints, feature_num is the feature dimension per node/joint.

    num_sample, _, _ = x.shape
    co_buffer = np.zeros((num_sample, time_step, node_num, feature_num))
    for i in range(num_sample):
        for j in range(time_step):
            buffer = np.reshape(x[i, j, :], (node_num, feature_num))
            co_buffer[i, j, :, :] = np.matmul(adj, buffer)

    return co_buffer


if __name__ == '__main__':

    results = []

    for person in range(1, 2):
        x_train, x_valid, y_train, y_valid = load_dance_data()

        time_step = 150
        body_num = 14
        feature_dim = 3
        gcn_units = 16
        lstm_units = 24
        class_num = 2

        adj_norm = make_graph()
        graph_train = combine(adj_norm, x_train, time_step, body_num, feature_dim)
        graph_valid = combine(adj_norm, x_valid, time_step, body_num, feature_dim)

        # model_name = 'gc-LSTM'
        callbacks = build_callbacks(person)

        model = build_model(time_step, body_num, feature_dim, gcn_units, lstm_units, class_num)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['binary_accuracy'])

        batch_size = 40
        epoch = 100

        model.fit(graph_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  callbacks=build_callbacks(person),
                  validation_data=(graph_valid, y_valid))

        print('---This is result for %s th subject---' % person)
        model.load_weights('Dancer' + str(person) + '.hdf5')
        y_pred = np.argmax(model.predict(graph_valid, batch_size=15), axis=1)
        y_true = np.argmax(y_valid, axis=1)
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
        class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None) * 100) * 0.01
        print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f1)))
        results.append(cf_matrix)

    with open('dance_results.txt', 'w') as f:
        for i in results:
            f.write(str(i))
