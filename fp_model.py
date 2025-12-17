"""
Script to Train and Test fp derivation model

@authors: E. De Leon, M. Vandevoorde, N. Gilet
"""

import sys

#!!! Append Path for CEFLIB !!!
sys.path.append('/home/machine_learning/CEFLIB/PYTHON')

import models
import preprocessing
import ceflib

import random
import numpy as np
from datetime import datetime
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Fix random seed 
random.seed(7)

# Filepahts
train_active_path ="../data_test/C3_CP_WHI_ACTIVE__20180401_000000_20180403_000000_V180621.cef"
train_electron_path="../data_test/C3_CP_WHI_ELECTRON_DENSITY__20180401_000000_20180403_000000_V190504.cef"

test_active_path="../data_test/C3_CP_WHI_ACTIVE__20200101_000000_20200102_000000_V200704.cef"
test_electron_path="../data_test/C3_CP_WHI_ELECTRON_DENSITY__20200101_000000_20200102_000000_V210204.cef"

# Read the Active Data
data={}
# Read the Active Data with CEFLIB
status = ceflib.read(train_active_path)
# Get the date from the Active file as datetime
data['date_act']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i)))for i in ceflib.var("time_tags")].copy()
# Get the whisper Active spectra
data['sptr_act'] = ceflib.var('Electric_Spectral_Power_Density').copy()
# Get the whisper Frequency table directly from the CEF
data['freq'] = ceflib.var("Spectral_Frequencies").copy()

# Read the electron density with CEFLIB
status = ceflib.read(train_electron_path)
data['date_dens']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i)))for i in ceflib.var("time_tags")].copy()
data['Electron_Density'] = ceflib.var('Electron_Density').copy()

# Pre-process Active data dB + Normalization
data['sptr_act'] = preprocessing.normSptr_db(data['sptr_act'],data['freq'])
data['sptr_act'] = data['sptr_act'][:,13:493]
# Change density to an index in the frequency table
fpe = preprocessing.calc_fpe_dens(data['Electron_Density'])
data['fp_index'] = [preprocessing.find_nearest_index(data['freq'],fp) for fp in fpe]
y = np.array(data['fp_index'])

# Select the active spectrum where the fpe index exists
x = np.empty((len(data["date_dens"]),480) )
for i,date in enumerate(data["date_dens"]):
    x[i]= data["sptr_act"][preprocessing.find_nearest_index(data['date_act'],date)]

# Split the datatest and reshape it to be usable by GRU
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.10)

x_train = x_train.reshape(-1,1,480)
x_test = x_test.reshape(-1,1,480)
y_train = y_train.reshape(-1,1,1)
y_test = y_test.reshape(-1,1,1)

# Change labels to categorical array classes
y_train = to_categorical(y_train, num_classes=480)

# Compile the fp (GRU) model
model = models.compile_fp_model_active()

# Train the model 
history = model.fit(x_train,y_train, epochs = 100, batch_size = 32, validation_split = 0.05) 

# Test on data from another day
data = {}
# Read the Active Data with CEFLIB
status = ceflib.read(test_active_path)
# Get the date from the Active file as datetime
data['date_act']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i)))for i in ceflib.var("time_tags")].copy()
# Get the whisper Active spectra
data['sptr_act'] = ceflib.var('Electric_Spectral_Power_Density').copy()
# Get the whisper Frequency table directly from the CEF
data['freq'] = ceflib.var("Spectral_Frequencies").copy()

# Read the electron density with CEFLIB
status = ceflib.read(test_electron_path)
data['date_dens']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i)))for i in ceflib.var("time_tags")].copy()
data['Electron_Density'] = ceflib.var('Electron_Density').copy()

# Pre-process Active data dB + Normalization
data['sptr_act'] = preprocessing.normSptr_db(data['sptr_act'],data['freq'])
data['sptr_act'] = data['sptr_act'][:,13:493]

# Reshape the data for GRU Model
x = data['sptr_act'].reshape(-1,1,480)   
# Classify with the GRU Model
classification = model.predict(x)
# Get the real fpe and the predicted Fpe
real_fpe = preprocessing.calc_fpe_dens(data['Electron_Density'])
predicted_fpe = [data['freq'][np.argmax(p)] for p in classification]

# Plot the results of prediction
plt.subplots(figsize=(20, 4))
plt.plot(data['date_dens'],real_fpe, label = "fpe")
plt.plot(data['date_act'],predicted_fpe, label = "predicted fpe")
plt.legend()
plt.savefig('result_fp.png')

# NOTE : Results will not be optimal has the trainning dataset 
# needs to be ~500k -1M sp and have a good distribution