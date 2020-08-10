# Decoding task states by spotting salient patterns at time points and brain regions

We proposed a novel architecture, Salient Patterns Over Time and Space (SPOTS), for decoding brain task states. 

In this git repository, we have added codes for two of the models that we tested:
- Baseline LSTM model
- SPOTS (our proposed model)

## Setup

`pip3 install -r requirements.txt`

## Guide to run the code

1. The data for the subjects resides in the `data` directory. You can create a subdirectory for a task name `data/task_*` and put the scans for each subject along with the labels in the task folder. We used the [HCP dataset](http://www.humanconnectomeproject.org). For your reference, we have added some dummy scans to give an idea of the format we expect the subject data in. The data should be a Numpy array for each subject, of the shape (V+1, T) where V is the number of regions of interest and T is the number of timepoints. The (V+1)th dimension contains the truth label for each time point. 

2. The codes reside in the `codes` folder. 
* `codes/lstm_only`: This folder contains the Python code for training an LSTM only RNN model. You can run the code by running the following command:
```
python3 train_model_lstm_only.py
```

* `codes/SPOT_model`: This folder contains the Python code for training the proposed RNN model built using Keras and Tensorflow. Attention was coded from [here](https://github.com/thushv89/attention_keras). You can run the code by running the following command:

```
python3 train_SPOT.py
```

## Attention Maps

Attention maps can be generated from the attention layer in SPOTS. This provides an interpretable visualisation to understand which time points are important in making a prediction. Below, we show how the attention maps change over epochs for window sizes 10, 20 and 30.

### Window Size: 10
![chunk-size-10](https://github.com/anonMiccai/SPOT/blob/master/window_size_10.gif)

### Window Size: 20
![chunk-size-20](https://github.com/anonMiccai/SPOT/blob/master/window_size_20.gif)

### Window Size: 30
![chunk-size-30](https://github.com/anonMiccai/SPOT/blob/master/window_size_30.gif)
