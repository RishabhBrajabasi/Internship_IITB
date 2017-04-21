#!/usr/bin/python

# -------------------------------------------------------------------------------
''' This script can be used for frame wise audio processing. It takes as input a
wav file and provides a framework for framewise processing of wav file.

Input argument: 1. Complete path of the wav file

Output : A csv file storing features extracted from wav file or enhanced wav file
	 Example for stroing the features/enhanced wav file is shown at the end
	 of the script.

Parameters: 1. Frame length (in milliseconds)
	        2. Hop length (in milliseconds)
	        3. Window type ('Hamming' by default)
	        4. Output file name (for storing features or enhanced wav file)

Author: Hitesh Tulsiani
Last updated: 9 Jan 2017
'''

# --------------------------------------------------------------------------------

from scipy.io import wavfile
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import hilbert

import wave


audio_file = 'C:\Users\Mahe\Desktop\Set 1\ABX_D_27122016_1_1.wav'  # Provide the absolute path of wav file as argument to script  (input argument)
window_dur = 30                                                    # Duration of window in millisec (Parameter 1)
hop_dur    = 10                                                    # Hop duration in millisec (Parameter 2)
fid_text   = open('C:\Users\Mahe\Desktop\eature_file.csv', 'w')    # Output file for storing features (Parameter 4)

fid_text_go   = open('C:\Users\Mahe\Desktop\ure_file.csv', 'w')    # Output file for storing features (Parameter 4)


fs, data = wavfile.read(audio_file)                                # Reading data from wav file in an array
                                                                   # Returns Sample rate and data read from  wav file
data = data / (float(2 ** 15))                                     # Normalizing it to [-1,1] range from [-2^15,2^15]

window_size = int(window_dur * fs * (0.001))                       # Converting window length to samples
hop_size    = int(hop_dur * fs * (0.001))                          # Converting hop length to samples

window_type = np.hamming(window_size)                              # Window type: Hamming (by default) (Parameter 3)

no_frames   = int(math.ceil(len(data) / (float(hop_size))))
zero_array  = np.zeros(window_size)
data        = np.concatenate((data, zero_array))

length=len(data)

ene = [0]*length
for j in range(length):
    ene[j] = data[j]*data[j]

fid_text_go.write("Data"+"\n")
for i in range(length):
    fid_text_go.write(str(ene[i]) + "\n")


#print " Window Size:",window_size," \n Hop Size:",hop_size, "\n No of frames:",no_frames

st_energy = [0] * no_frames                                        # Create arrays which are empty
maximum = [0] * no_frames

#fileNameTemplate = r'C:\Users\Mahe\Desktop\Hopethisworks\Plot{0:02d}.png'


for i in range(no_frames):

    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    #print i*hop_size,i*hop_size+window_size,len(frame)
    #print data[i * hop_size:i * hop_size + window_size] * window_type
    #so frame is an array with 480 data points, stored in a 1D array
    #for j in range(i*hop_size,i*hop_size + window_size):



    # Do frame wise processing here

    # ---------- For example: Computing short time energy -------------------
    st_energy[i] = sum(frame ** 2)
    maximum[i]=max(frame)


    ## Code to individually plot each of the frames created. Since, 4097 plots will be created, run it sparingly
    #plt.plot(frame)
    #plt.savefig(fileNameTemplate.format(i), format='png')
    #plt.clf()

# ------------------------------------------------------------------------

# ------------------------------




# After processing, you can either store the features extracted or
# the enhanced wavefile

# Storing extracted features in file (As an example, short time energy is stored using command below)
fid_text.write("Frames" + "," + "Short term energy" + "," + "Maximum" + "\n")
for i in range(no_frames):
    fid_text.write(str(i + 1) + "," + str(st_energy[i]) + "," + str(maximum[i]) + "\n")



#analytic_signal = hilbert(st_energy)
#amplitude_envelope = np.abs(analytic_signal)

plt.figure(1)

plt.subplot(311)
plt.plot(data,'g')

plt.subplot(312)
plt.plot(ene,'r')

#plt.subplot(313)
#plt.plot(maximum,'b')

plt.show()
# Storing enhanced wavefile (Just to show input data array is written as output wav file)
wavfile.write("C:\Users\Mahe\Desktop\output.wav", fs, data)

