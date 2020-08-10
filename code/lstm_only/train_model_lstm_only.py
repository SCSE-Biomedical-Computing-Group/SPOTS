## Modules
import os
os.environ['PYTHONHASHSEED'] = '0'

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

import pickle
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import sys

def model(x_train, num_labels, units, num_lstm_layers, model_type = 'lstm'):
    """
    Simple RNN Model with multiple RNN layers and units.
    Inputs:
    - x_train: required for creating input shape for RNN layer in Keras
    - num_labels: number of output classes (int)
    - units: number of RNN units (float)
    - num_lstm_layers: number of RNN layers to add (int)
    - model_type: Type of RNN model (str)
    Returns
    - model: A Keras model
    """
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])) if model_type == 'lstm'  else
            GRU(units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])) if model_type == 'gru' else
            SimpleRNN(units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        ] +
        [
            LSTM(units, return_sequences=True) if model_type == 'lstm' else 
            GRU(units, return_sequences=True) if model_type == 'gru' else
            SimpleRNN(units, return_sequences=True),
        ] * (num_lstm_layers - 1) +
        [
            Dense(num_labels, activation = 'softmax')
        ])

    print (model.summary())

    return model

def get_data(choice_task, src_dir, chunk_size):
	"""
    Generates the data for task fMRI data stored at src_dir/choice_task

    Inputs:
    - choice_task: name of the task folder stored in src_dir where you wish to get your data from (str)
    - src_dir: base directory where your task data folders are located (str)
    - chunk_size: size of the time windows that you want to extract from each task fMRI scans. Total number of windows from each fMRI scan (approx): number of time points/chunk_size (int)
    Returns
    - model: A Keras model
    """
	data_path = os.path.join(src_dir, choice_task)
	subject_names_list = []
	all_matrices = []
	all_matrices_labels = []

	for i, file_or_dir in enumerate(os.listdir(data_path)):
		matrix = np.load(os.path.join(data_path, file_or_dir))
		matrix = np.nan_to_num(matrix)
		num_time_points = np.array(matrix).shape[1]
		labels = matrix[-1]

		chunk_num = 0
		for start, end in zip(range(0, num_time_points, chunk_size), range(chunk_size, num_time_points, chunk_size)):
			chunk_labels = labels[start:end]
			chunk_matrix = matrix[0:-1, start:end]
			# print (matrix.shape, np.min(matrix, axis=1).shape, np.max(matrix, axis=1).shape)
			all_matrices.append(chunk_matrix.T)
			all_matrices_labels.append(chunk_labels)
			subject_names_list.append(file_or_dir[-15:-6] ) #+ '_' + str(chunk_num)
			chunk_num += 1


	num_labels = np.max(all_matrices_labels) + 1
	num_samples = np.array(all_matrices).shape[0]

	return (subject_names_list, np.array(all_matrices), np.array(all_matrices_labels))

def convert_to_one_hot(y_labels, num_labels):
	'''
	Convert labels matrix y_labels (shape number of samples X num time points) containing total num_labels to a one hot array. 
	Inputs:
	y_labels: numpy matrix specifying labels for each sample and each time point. (numpy array, shape (number of samples, num time points))
	num_labels: total number of classes.

	Returns:
	One hot numpy array of shape (num_samples, num_time_points, num_labels) 

	'''
	num_samples = np.array(y_labels).shape[0]
	num_time_points = np.array(y_labels).shape[1]

	y_one_hot_labels = np.zeros((num_samples, num_time_points, num_labels))

	for i_sample, sample_time_points in enumerate(y_labels):
		for i_time, time_point in enumerate(sample_time_points):
			y_one_hot_labels[i_sample, i_time, int(y_labels[i_sample, i_time])] = 1

	return y_one_hot_labels

## Constants 
CHUNK = 40
EPOCH = 200
BATCH_SIZE = 4
LSTM_UNITS = 512
NUM_LSTM_LAYERS = 1
LEARNING_RATE = 0.05
SAVE_DIR = './'
PATIENCE = 10
SEED = 0
MODEL_TO_RUN = 'lstm' # gru lstm ; must be in small letters
TASK = 'LANGUAGE_LR'

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

if __name__ == '__main__':
	results_ground_truth = []
	early_stopping_epoch_list = []	

	# Prepare dataset
	(subject_names_list, data, labels) = get_data(TASK, '../../data/', CHUNK)
	NUM_LABELS = int(np.max(labels) + 1)

	num_labels = np.max(labels, axis = None) + 1

	print("Dataset Dimensions" + str(data.shape))

	subjects_train, subjects_test, x_train, x_test, y_train, y_test = train_test_split(subject_names_list, data, labels, test_size=0.15, random_state=SEED, shuffle = False)
	subjects_train, subjects_test, x_train, x_test, y_train, y_test = np.array(subjects_train), np.array(subjects_test), np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

	print ('Train Subjects', subjects_train.shape, 'Train data shape', x_train.shape, 'Train labels shape', y_train.shape)
	print ('Test Subjects', subjects_test.shape, 'Test data shape', x_test.shape,'Test labels shape', y_test.shape)
	print ('Number of classes', NUM_LABELS)

	y_train_one_hot, y_test_one_hot = convert_to_one_hot(y_train, NUM_LABELS), convert_to_one_hot(y_test, NUM_LABELS)

	scalers = {}
	for i in range(x_train.shape[2]):
	    scalers[i] = StandardScaler()#MinMaxScaler(feature_range=(0, 1))
	    x_train[:, :, i] = scalers[i].fit_transform(x_train[:, :, i])
	    x_test[:, :, i] = scalers[i].transform(x_test[:, :, i])

	# Train model
	rnn_model = model(x_train, NUM_LABELS, LSTM_UNITS, NUM_LSTM_LAYERS, MODEL_TO_RUN)

	model_filename = SAVE_DIR + '/best_model_lstm_only_seed_' + str(SEED) + '_chunk_' + str(CHUNK) + '.h5'
	callbacks = [ModelCheckpoint(filepath=model_filename, monitor = 'val_acc', save_weights_only=True, save_best_only=True), EarlyStopping(monitor='val_acc', patience=PATIENCE)]#, LearningRateScheduler()]

	opt = optimizers.Adam(clipnorm=1.)#(lr=LEARNING_RATE, clipnorm=1., decay=LEARNING_RATE/EPOCH)

	rnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

	train_trailing_samples =  x_train.shape[0]%BATCH_SIZE
	test_trailing_samples =  x_test.shape[0]%BATCH_SIZE

	print (train_trailing_samples, test_trailing_samples)

	x_train_ = x_train[0:-train_trailing_samples]
	x_test_ = x_test[0:-test_trailing_samples]

	y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
	y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]


	print ('Train data shape', x_train_.shape, 'Train labels shape', y_train_one_hot.shape)
	print ('Test data shape', x_test_.shape,'Test labels shape', y_test_one_hot.shape)

	history = rnn_model.fit(x_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(x_test_, y_test_one_hot))

	early_stopping_epoch = callbacks[1].stopped_epoch - PATIENCE + 1 # keras gives the 0-index value of the epoch, so +1
	print('Early stopping epoch: ' + str(early_stopping_epoch))
	early_stopping_epoch_list.append(early_stopping_epoch)

	if early_stopping_epoch < 0:
	    early_stopping_epoch = 100

	# Evaluate model and predict data on TEST 
	print("******Evaluating TEST set*********")
	scores_test = rnn_model.evaluate(x_test_, y_test_one_hot)
	print("model: \n%s: %.2f%%" % (rnn_model.metrics_names[1], scores_test[1]*100))

	y_test_predict = rnn_model.predict_classes(x_test_)
	y_test_predict = np.array(y_test_predict)
	y_test = np.array(y_test[0:-test_trailing_samples])

	all_trainable_count = int(np.sum([K.count_params(p) for p in set(rnn_model.trainable_weights)]))

	MAE = metrics.mean_absolute_error(y_test.flatten(), y_test_predict.flatten(), sample_weight=None, multioutput='uniform_average')

	with open(SAVE_DIR + '/results_test_' + MODEL_TO_RUN + '.csv', 'a') as out_stream:
	    out_stream.write(str(CHUNK) + ',' + str(SEED) + ',' + str(early_stopping_epoch) + ',' + str(all_trainable_count) + ',' + str(scores_test[1]*100) + ',' + str(scores_train[1]*100) + ',' + str(MAE) + '\n')

	K.clear_session()
