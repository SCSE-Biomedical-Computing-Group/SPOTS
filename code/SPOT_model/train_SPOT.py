## Modules
import os
os.environ['PYTHONHASHSEED'] = '0'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True

import pickle
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt
from pylab import savefig
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Input, Lambda # , dot, Activation, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend as K
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from model_with_cnn import define_attn_model_w_cnn
from tqdm import tqdm
import sys

def plot_attn_weights(CHUNK, attn_weights, y_predict, y_label, sample_id, epoch, mode = 0):
	"""
	Plots the attention weight for a sample. 
	Inputs:
	- CHUNK (int): the size of each time window
	- attn_weights (numpy array): the attention weight array with matrices for each sample.
	- y_predict (list): the list of predicted labels for all samples and time points.
	- y_label (list): the list of ground-truth labels for all samples and time points.
	- sample_id (int): the sample which you want to plot the attention weights. 
	- epoch (int): the epoch for which the attention weights are being computed 
	- mode (int): ignore
	Return:
		None
	"""
	plt.figure()
	plt.imshow(attn_weights[sample_id])
	plt.colorbar()
	plt.xticks(np.arange(CHUNK), y_predict[sample_id*CHUNK:(sample_id+1)*CHUNK])
	plt.xlabel('Predicted Labels')
	y_true_ = y_label.flatten(order = 'C')[sample_id*CHUNK:(sample_id+1)*CHUNK]
	plt.yticks(np.arange(CHUNK), y_true_.astype(int))
	plt.ylabel('True Labels')
	plt.clim(0, 1)
	if not os.path.exists(os.path.join(SAVE_DIR, 'results')):
		os.mkdir(os.path.join(SAVE_DIR, 'results'))
	plt.savefig(os.path.join(SAVE_DIR, 'results', str(TASK) + '_attention_with_cnn_MODE_' + str(mode) + '_CHUNK_' + str(CHUNK) + '_sample_' + str(sample_id) + '_epoch_' + str(epoch) + '.png'))
	plt.close()


def infer_ts_w_cnn(encoder_model, decoder_model, test_seq, num_labels, chunk_size):
	"""
	Infer logic, used for testing and getting the attention weights.
	Inputs:
	-  encoder_model: keras.Model
	-  decoder_model: keras.Model
	-  test_seq: test sequence containing the brain ROI time series.
	-  num_labels: int
	-  chunk_size: int
	Returns:
	- test_output: The labels assigned to each test sample and its time points.
	- attention_weights: The attention weights assigned to each test sample and its time points.
	"""
	attention_weights = []
	test_output = []

	for sample in range(0, test_seq.shape[0]):
		test_sample = np.reshape(test_seq[sample, :,], (1, test_seq.shape[1], -1, 1))
		enc_outs, enc_last_state_h, enc_last_state_c = encoder_model.predict(test_sample)
		dec_state_h, dec_state_c = enc_last_state_h, enc_last_state_c
		dec_test_one_hot = np.random.normal(loc=NOISE_MEAN, scale=NOISE_STD, size = (1, 1, num_labels))
		
		for i in range(chunk_size):
			dec_out, attention, dec_state_h, dec_state_c = decoder_model.predict([enc_outs, dec_state_h, dec_state_c, dec_test_one_hot])  #[encoder_inf_states, decoder_init_state, decoder_inf_inputs]  # first decoder_inf_inputs is random input of size (batch, 1, num_classes)
			dec_out = dec_out[0, 0, :]
			attention = attention[0, 0, :]
			dec_ind = np.argmax(dec_out)

			attention_weights.append(attention)
			test_output.append(dec_ind)

	return np.array(test_output), np.array(attention_weights)

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
			all_matrices.append(chunk_matrix.T)
			all_matrices_labels.append(chunk_labels)
			subject_names_list.append(file_or_dir[-15:-6] )
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
CHUNK =  40
EPOCH = 500
BATCH_SIZE = 4
LSTM_UNITS = 32
CNN_FILTERS = 4
LEARNING_RATE = 0.05
SAVE_DIR = './'
PATIENCE = 10
NOISE_STD = 0.2
NOISE_MEAN = 0.4
SEED = 0

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ATTN_LAYER_MODE = 0
TASK = 'LANGUAGE_LR'


if __name__ == '__main__':
	(subject_names_list, data, labels) = get_data(TASK, '../../data/', CHUNK)
	NUM_LABELS = int(np.max(labels) + 1)

	print('Distribution of labels: ', np.unique(labels, return_counts = True))

	best_test_acc = 0
	intial_epoch =0
	num_epoch_no_improvement = 0

	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)

	# Prepare dataset
	subjects_train, subjects_test, x_train, x_test, y_train, y_test = train_test_split(subject_names_list, data, labels, test_size=0.15, random_state=SEED, shuffle = False)
	subjects_train, subjects_test, x_train, x_test, y_train, y_test = np.array(subjects_train), np.array(subjects_test), np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

	print ('Train Subjects', subjects_train.shape, 'Train data shape', x_train.shape, 'Train labels shape', y_train.shape)
	print ('Test Subjects', subjects_test.shape, 'Test data shape', x_test.shape,'Test labels shape', y_test.shape)

	scalers = {}
	for i in range(x_train.shape[2]):
		scalers[i] = StandardScaler() #MinMaxScaler(feature_range=(0, 1))
		x_train[:, :, i] = scalers[i].fit_transform(x_train[:, :, i])
		x_test[:, :, i] = scalers[i].transform(x_test[:, :, i])


	full_model, infer_enc_model, infer_dec_model, attn_layer, attn_states = define_attn_model_w_cnn(num_filters = CNN_FILTERS, hidden_size=LSTM_UNITS, batch_size=BATCH_SIZE, en_timesteps=CHUNK, fr_timesteps=CHUNK, en_vsize=264, fr_vsize=NUM_LABELS, attn_layer_type=ATTN_LAYER_MODE)
	
	for ep in range(EPOCH):
		
		y_train_one_hot, y_test_one_hot = convert_to_one_hot(y_train, NUM_LABELS), convert_to_one_hot(y_test, NUM_LABELS)
		losses, accuracies = [], []

		for batch_index, bi in enumerate(range(0, x_train.shape[0] - BATCH_SIZE, BATCH_SIZE)):

			input_batch = np.expand_dims(x_train[bi:bi + BATCH_SIZE, :, :], axis = 3)

			#s = np.zeros((BATCH_SIZE, 1, NUM_LABELS))
			NOISE = np.random.normal(loc=NOISE_MEAN, scale=NOISE_STD, size = y_train_one_hot[bi:bi + BATCH_SIZE].shape)
			#y_train_one_hot_batch_decoder =  y_train_one_hot[bi:bi + BATCH_SIZE] +  NOISE #np.concatenate((s,  y_train_one_hot[bi:bi + BATCH_SIZE, 0:-1]), axis = 1)
			y_train_one_hot_batch_decoder = np.zeros(y_train_one_hot[bi:bi + BATCH_SIZE].shape) + NOISE
			y_train_one_hot_batch_out = y_train_one_hot[bi:bi + BATCH_SIZE]
			full_model.train_on_batch([input_batch, y_train_one_hot_batch_decoder], y_train_one_hot_batch_out)
			l = full_model.evaluate([input_batch, y_train_one_hot_batch_decoder], y_train_one_hot_batch_out, batch_size=BATCH_SIZE, verbose=0)

			if batch_index%10 == 0:
				print('Epoch:', ep, 'Batch', batch_index, 'Loss', l[0], 'Accuracy', l[1])

			losses.append(l[0])
			accuracies.append(l[1])
		
		if (ep + 1) %10 != 0: #don't compute test accuracy for all samples at every epoch.
			x_test_, y_test_ = x_test[0:100], y_test[0:100]
		else:
			x_test_, y_test_ = x_test, y_test

		y_test_predict, attn_weights = infer_ts_w_cnn(infer_enc_model, infer_dec_model, x_test_, NUM_LABELS, CHUNK)
		attn_weights_reshaped = np.reshape(attn_weights, (-1, CHUNK, CHUNK))
		plot_attn_weights(CHUNK, attn_weights_reshaped, y_test_predict, y_test_, 0, ep, ATTN_LAYER_MODE)

		y_test_predict = np.array(y_test_predict)
		y_test_ = np.array(y_test_)

		y_test_flattened = y_test_.flatten(order = 'C')
		cm = confusion_matrix(y_test_flattened, y_test_predict)
		test_acc = np.trace(cm)/np.sum(cm)
		MAE = metrics.mean_absolute_error(y_test_.flatten(), y_test_predict.flatten(), sample_weight=None, multioutput='uniform_average')
		

		scores_train = [np.mean(losses), np.mean(accuracies)]
		print("Epoch %.2f Train model: \n%s: %.2f%%" % (ep, full_model.metrics_names[1], scores_train[1]*100))
		
		all_trainable_count = int(np.sum([K.count_params(p) for p in set(full_model.trainable_weights)]))
		print("Epoch %.2f Test model: \n%s: %.2f%%" % (ep, full_model.metrics_names[1], test_acc*100))

		if test_acc >= best_test_acc:
			print("Test accuracy increases from {:.4f} to {:.4f}".format(best_test_acc, test_acc))
			best_test_acc = test_acc
			num_epoch_no_improvement = 0
			#save model
			model_filename = 'best_model_' + str(TASK) + '_SPOT.h5'
			full_model.save(os.path.join(SAVE_DIR, model_filename))
			print("Saving model ", os.path.join(SAVE_DIR, model_filename))

		else:
			num_epoch_no_improvement += 1
			print("Test accuracy does not increase from {:.4f}, num_epoch_no_improvement {}".format(best_test_acc, num_epoch_no_improvement))

		if num_epoch_no_improvement == PATIENCE:
			print("Early Stopping")
			early_stopping_epoch = ep - PATIENCE
			break
		
		early_stopping_epoch = ep - num_epoch_no_improvement

		with open(SAVE_DIR + '/results_' + str(TASK) + '_SPOT.csv', 'a') as out_stream:
			out_stream.write(str(CHUNK) + ',' + str(SEED) + ',' + str(ep) + ',' + str(all_trainable_count) + ',' + str(test_acc*100) + ',' + str(scores_train[1]*100) + '\n')

	K.clear_session()
