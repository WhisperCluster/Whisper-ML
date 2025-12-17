"""
Script to Train and Test Natural selection model

@authors: E. De Leon, M. Vandevoorde, N. Gilet
"""

import sys

#!!! Append Path for CEFLIB !!!
sys.path.append('/home/machine_learning/CEFLIB/PYTHON')

import random
import models
import preprocessing
import ceflib
import numpy as np
from datetime import datetime
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Fix random seed 
random.seed(7)

# Filepaths
train_natural_path = "../data_test/C3_CP_WHI_NATURAL__20180401_000000_20180403_000000_V180621.cef"
test_natural_path = "../data_test/C3_CP_WHI_NATURAL__20200101_000000_20200102_000000_V200704.cef"

data={}
# Read the Natural Data with CEFLIB
status = ceflib.read(train_natural_path)
# Get the date from the Natural file as datetime
data['date_nat']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i))) for i in ceflib.var("time_tags")].copy()
# Get the whisper Natural spectra
data['sptr_nat'] = ceflib.var('Electric_Spectral_Power_Density').copy()

# Get the whisper Frequency table directly from the CEF
data['freq'] = ceflib.var("Spectral_Frequencies")

# Pre-process Natural data dB + Normalization
data['sptr_nat'] = preprocessing.normSptr_db(data['sptr_nat'],data['freq'])

# Get only the 470 used bins in this example
x = data['sptr_nat'][:, 14:484]

# Create region labels manually using datetime 0: Other, 1: Plasmasphere
y = np.zeros(len(data['date_nat']))

# Plasmasphere signature indexes based on time, format YYYY,MM,D,HH,MM
idx = np.where((data['date_nat']>= np.datetime64("2018-04-01T12:30:00")) & (data['date_nat']< np.datetime64("2018-04-02T01:00:00")))[0]
y[idx]=1

# Change labels to categorical array classes
y=to_categorical(y, num_classes=2)

# Split the train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Compiling PS model, Same for trainning the TL model 
model = models.compile_ps_model_natural()

# Train the de model 
history=model.fit(x_train, y_train, epochs=5, batch_size=16,validation_split=0.05)

# Test the model on another day
data={}
# Read the Natural Data with CEFLIB
status = ceflib.read(test_natural_path)

# Get the date from the Natural file as datetime
data['date_nat']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i))) for i in ceflib.var("time_tags")].copy()
# Get the whisper Natural spectra
data['sptr_nat'] = ceflib.var('Electric_Spectral_Power_Density').copy()

# Get the whisper Frequency table directly from the CEF
data['freq'] = ceflib.var("Spectral_Frequencies")

# Pre-process Natural data dB + Normalization
data['sptr_nat'] = preprocessing.normSptr_db(data['sptr_nat'],data['freq'])
# Get only the 470 used bins in this example
data['sptr_nat'] = data['sptr_nat'][:, 14:484]

# Predict with PS model
proba = model.predict(data['sptr_nat'])

# Plot the results of prediction
plt.subplots(figsize=(20, 4))
plt.plot(data['date_nat'],proba[:,0])
plt.savefig('result_natural.png')


# NOTE : Results will not be optimal has the trainning dataset 
# needs to be ~500k sp and have a good distribution