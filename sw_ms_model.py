
import sys
sys.path.append('/home/machine_learning/CEFLIB/PYTHON')

import models
import preprocessing
import ceflib
import numpy as np
from datetime import datetime
from keras.utils import plot_model, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data={}
# Read the Natural Data with CEFLIB
status = ceflib.read("/home/natural_pipeline/archive_github/data_test/C3_CP_WHI_NATURAL__20180401_000000_20180403_000000_V180621.cef")

# Get the date from the Natural file as datetime
data['date_nat']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i))) for i in ceflib.var("time_tags")].copy()
# Get the whisper Natural spectra
data['sptr_nat'] = ceflib.var('Electric_Spectral_Power_Density').copy()

# Get the whisper Frequency table directly from the CEF
data['freq'] = ceflib.var("Spectral_Frequencies")

# Read the Active Data with CEFLIB
status = ceflib.read("/home/natural_pipeline/archive_github/data_test/C3_CP_WHI_ACTIVE__20180401_000000_20180403_000000_V180621.cef")
# Get the date from the Active file as datetime
data['date_act']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i)))for i in ceflib.var("time_tags")].copy()
# Get the whisper Active spectra
data['sptr_act'] = ceflib.var('Electric_Spectral_Power_Density')

# Pre-process Natural data dB + Normalization
data['sptr_nat'] = preprocessing.normSptr_db(data['sptr_nat'],data['freq'])
# Pre-process Active data dB + Normalization
data['sptr_act'] = preprocessing.normSptr_db(data['sptr_act'],data['freq'])

# Get only the 480 used bins in this example
data['sptr_nat'] = data['sptr_nat'][:, 14:484]
# Get only the 480 used bins in this example
data['sptr_act'] = data['sptr_act'][:,13:493]

# Concatenate the active and natural closest spectrum
data['sptr'] = np.zeros((len(data['sptr_act']),950))
for i in range(0,len(data['sptr_act'])):
    idx = preprocessing.find_nearest_index(data['date_nat'],data['date_act'][i])
    data['sptr'][i]= np.concatenate((data['sptr_act'][i],data['sptr_nat'][idx]),axis=0)

# Create region labels manually using datetime 0: Other, 1: Magnetosheath, 2: Solar Wind
y = np.zeros(len(data['date_act']))

# Magnetosheath indexes based on time, format YYYY,MM,D,HH,MM
idx = np.where((data['date_act']>= np.datetime64("2018-04-01T05:30:00")) & (data['date_act']< np.datetime64("2018-04-01T11:00:00")))[0]
y[idx]=1
idx = np.where((data['date_act']>= np.datetime64("2018-04-02T07:25:00")) & (data['date_act']< np.datetime64("2018-04-02T17:55:00.")))[0]
y[idx]=1

# Solar wind indexes based on time, format YYYY,MM,D,HH,MM
idx = np.where((data['date_act']>= np.datetime64("2018-04-01T00:00:00")) & (data['date_act']< np.datetime64("2018-04-01T05:30:00")))[0]
y[idx]=2
idx = np.where((data['date_act']>= np.datetime64("2018-04-02T07:55:00")) & (data['date_act']< np.datetime64("2018-04-02T23:59:59.999")))[0]
y[idx]=2

# Change labels to categorical array classes
y=to_categorical(y, num_classes=3, dtype='float32')
x = data['sptr']

# Split the train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

# Compile the model
model = models.compile_sw_ms_model_active_natural()

# Train the model, change epochs and validation split if needed
history = model.fit(x_train, y_train, epochs=32, batch_size=32,validation_split=0.15)


# Test the model on another day
data={}
# Read the Natural Data with CEFLIB
status = ceflib.read("/home/natural_pipeline/archive_github/data_test/C3_CP_WHI_NATURAL__20200101_000000_20200102_000000_V200704.cef")

# Get the date from the Natural file as datetime
data['date_nat']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i))) for i in ceflib.var("time_tags")].copy()
# Get the whisper Natural spectra
data['sptr_nat'] = ceflib.var('Electric_Spectral_Power_Density').copy()

# Get the whisper Frequency table directly from the CEF
data['freq'] = ceflib.var("Spectral_Frequencies")

# Read the Active Data with CEFLIB
status = ceflib.read("/home/natural_pipeline/archive_github/data_test/C3_CP_WHI_ACTIVE__20200101_000000_20200102_000000_V200704.cef")
# Get the date from the Active file as datetime
data['date_act']= [np.datetime64(datetime.utcfromtimestamp(ceflib.milli_to_timestamp(i)))for i in ceflib.var("time_tags")].copy()
# Get the whisper Active spectra
data['sptr_act'] = ceflib.var('Electric_Spectral_Power_Density')

# Pre-process Natural data dB + Normalization
data['sptr_nat'] = preprocessing.normSptr_db(data['sptr_nat'],data['freq'])
# Pre-process Active data dB + Normalization
data['sptr_act'] = preprocessing.normSptr_db(data['sptr_act'],data['freq'])

# Get only the 480 used bins in this example
data['sptr_nat'] = data['sptr_nat'][:, 14:484]
# Get only the 480 used bins in this example
data['sptr_act'] = data['sptr_act'][:,13:493]

# Concatenate the active and natural closest spectrum
data['sptr'] = np.zeros((len(data['sptr_act']),950))
for i in range(0,len(data['sptr_act'])):
    idx = preprocessing.find_nearest_index(data['date_nat'],data['date_act'][i])
    data['sptr'][i]= np.concatenate((data['sptr_act'][i],data['sptr_nat'][idx]),axis=0)

proba = model.predict(data['sptr'])

fig = plt.figure()
plt.subplots(figsize=(20, 4))
plt.plot(data['date_act'],proba)
fig.savefig('result.png', dpi=fig.dpi)