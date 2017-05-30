"""
Envelope_Extension.py
Applying rule based techniques to the envelope of the audio files.
"""
from __future__ import division

import csv
import glob
import os
import re
import sys
import win32api
import winsound
from datetime import datetime
from shutil import copyfile
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy.signal import hilbert

startTime = datetime.now()  # To calculate the run time of the code.
#----------------------------------------------------------------------------------------------------------------------#
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
#----------------------------------------------------------------------------------------------------------------------#
def butter_low_pass(low_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    b, a = butter(order, low, btype='low')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_low_pass_filter(data_low, lowcut, fs, order=5):
    b, a = butter_low_pass(lowcut, fs, order=order)
    y = lfilter(b, a, data_low)
    return y
#----------------------------------------------------------------------------------------------------------------------#
def butter_band_pass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_band_pass_filter(data_band, lowcut, highcut, fs, order=5):
    b, a = butter_band_pass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data_band)
    return y
#----------------------------------------------------------------------------------------------------------------------#

file_no = '17'
audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + '.wav'
window_dur = 30
hop_dur = 5
fs, audio_data = wavfile.read(audio_file)  # Extract the sampling frequency and the data points of the audio file.
audio_data = audio_data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
audio_data = butter_band_pass_filter(audio_data, 300, 2500, fs, order=6)  # Filtering the data.
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(audio_data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
audio_data = np.concatenate((audio_data, zero_array))

st_energy = []
for i in range(no_frames):  # Calculating frame wise short term energy
    frame = audio_data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
    st_energy.append(sum(frame ** 2))  # Calculating the short term energy

st_energy = moving_average(st_energy, 5)
max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
for i in range(no_frames):
    st_energy[i] = st_energy[i]/max_st_energy  # Normalizing the curve
st_energy = st_energy.tolist()
#----------------------------------------------------------------------------------------------------------------------#
"""
Find all maxima's in the envelope of the audio file. Maxima's which have a magnitude above the threshold are
marked as peaks. The magnitude of the peak and the location of the peak is stored.
"""
threshold_peak = 0.05  # Minimum value for a maxima to qualify as a peak.
peak_1 = []
location_peak_1 = []
for p in range(len(st_energy)):
    if p == 0:
        peak_1.append(0)
    elif p == len(st_energy) - 1:
        peak_1.append(0)
    else:
        if st_energy[p] > st_energy[p + 1] and st_energy[p] > st_energy[p - 1] and st_energy[p] >= threshold_peak:
            peak_1.append(st_energy[p])  # Finding value of the peak.
            location_peak_1.append(p)  # Finding location of the peak.
        else:
            peak_1.append(0)
#----------------------------------------------------------------------------------------------------------------------#
"""
Find flash points[called flash points after the DC comics character flash]. They indicate points on the envelope
of the audio file which lie at magnitude og 0.85 times the magnitude of the peak. There is one such point to the
peak and one such point to the right of the peak. If any other peaks lie within the region between these two flash
points then they are removed.

Format of 'the_list_1' :
Location of flash point to the left of the peak || Location of peak || Location of flash point to the right of the peak || tracker variable || Magnitude of peak
"""
the_list_1 = [[]]
count = 0
for l1 in range(len(location_peak_1)):
    value = st_energy[location_peak_1[l1]] * 0.85
    for ele in range(location_peak_1[l1], -1, -1):
        if st_energy[ele] < value:
            flash_1 = ele  # Finding the first flash point, to the left of the peak.
            break
    for ele in range(location_peak_1[l1], len(st_energy), 1):
        if st_energy[ele] < value:
            flash_2 = ele  # Finding the second flash point, to the right of the peak.
            break
    the_list_1.append([flash_1, location_peak_1[l1], flash_2, count, st_energy[location_peak_1[l1]]])
    count += 1
the_list_1.pop(0)

# print the_list_1

descending_order_of_peaks = sorted(the_list_1, key=lambda student: student[4], reverse=1)

remove = []  # List of peaks which are to be removed.
for l in range(len(descending_order_of_peaks)):
    for k in range(l + 1, len(descending_order_of_peaks)):
        if descending_order_of_peaks[l][0] < descending_order_of_peaks[k][1] < descending_order_of_peaks[l][2] and descending_order_of_peaks[k][3] not in remove:
            remove.append(descending_order_of_peaks[k][3])

the_list_2 = [[]]  # Shortlisted peak candidates
for iteration in the_list_1:
    if iteration[3] not in remove:
        the_list_2.append(iteration)
the_list_2.pop(0)


# #----------------------------------------------------------------------------------------------------------------------#
"""
Between a pair of two peaks, the point with the least magnitude is found and is marked as a valley.
The staring point and the last point of the envelope are also marked as valleys. If the valley is located within
the region of the flash points of the peak, the peak is removed. Once the shortlisted peaks are obtained, valleys
between peaks are found again as mentioned earlier.
"""
valley_1 = [0]
for v1 in range(len(the_list_2) - 1):
    minimum = min(st_energy[the_list_2[v1][1]:the_list_2[v1 + 1][1]])  # Region between the two peaks
    valley_1.append(st_energy.index(minimum))
valley_1.append(len(st_energy) - 1)

the_list_3 = [[]]
fresh_count = 0  # Tracker variable to aid in removing peaks later.
for l3 in range(len(the_list_2)):
    the_list_3.append([valley_1[l3], the_list_2[l3][0], the_list_2[l3][1], the_list_2[l3][2], valley_1[l3 + 1], fresh_count])
    fresh_count += 1
the_list_3.pop(0)

remove[:] = []
for cc in the_list_3:
    if cc[3] > cc[4] or cc[0] > cc[1]:
        remove.append(cc[5])

peak_2 = []
for sf in the_list_3:
    if sf[5] not in remove:
        peak_2.append(sf[2])

valley_2 = [0]
for v in range(len(peak_2) - 1):
    minimum = min(st_energy[peak_2[v]:peak_2[v + 1]])
    valley_2.append(st_energy.index(minimum))
valley_2.append(len(st_energy) - 1)
#----------------------------------------------------------------------------------------------------------------------#
"""
Re-calculating flash points for the remaining peaks. Using flash point and valley information to determine
suitable boundary for start and end of vowel. Flash points are calculated as points having a magnitude of
0.707 times the magnitude of the peak.

Format of 'the_list_3':
Location of valley to the left of peak||Location of flash point to the left of the peak||Location of Peak||
Location of flash point to the right of the peak||Location of the valley to the right of the peak
"""
the_list_4 = [[]]
for l3 in range(len(peak_2)):
    value = st_energy[peak_2[l3]] * 0.707
    for ele in range(peak_2[l3], -1, -1):
        if st_energy[ele] < value:
            flash_1 = ele
            break
    for ele in range(peak_2[l3], len(st_energy), 1):
        if st_energy[ele] < value:
            flash_2 = ele
            break
    the_list_4.append([valley_2[l3], flash_1, peak_2[l3], flash_2, valley_2[l3 + 1]])
the_list_4.pop(0)

print len(the_list_1), len(the_list_2), len(the_list_3), len(the_list_4)

# import matplotlib.pyplot as plt
# plt.plot(st_energy)
# for i in range(len(the_list_4)):
#     plt.vlines(the_list_4[i][2], 0, st_energy[the_list_4[i][2]])
# plt.show()

boundary = [[]]
for b in the_list_4:
    if b[0] < b[1] < b[3] < b[4]:  # valley_left < flash_left < flash_right < valley_right
        boundary.append([b[1], b[3]])  # flash_left <--> flash_right

    elif b[0] < b[1] and b[3] > b[4]:  # valley_left < flash_left and flash_right > valley_right
        boundary.append([b[1], b[4]])  # flash_left <--> valley_right

    elif b[0] > b[1] and b[3] < b[4]:  # valley_left > flash_left and flash_right < valley_right
        boundary.append([b[0], b[3]])  # valley_left <--> flash_right

    elif b[0] > b[1] and b[3] > b[4]:  # valley_left < flash_left and flash_right > valley_right
        boundary.append([b[0], b[4]])  # valley_left <--> valley_right
    else:
        boundary.append([b[0], b[4]])  # valley_left <--> valley_right
boundary.pop(0)

# print the_list_4
# print boundary

# import matplotlib.pyplot as plt
# plt.plot(st_energy)
# for i in range(len(the_list_4)):
#     plt.vlines(the_list_4[i][2], 0, st_energy[the_list_4[i][2]])
#     plt.vlines(boundary[i][0], 0, st_energy[boundary[i][0]], color='red')
#     plt.vlines(boundary[i][1], 0, st_energy[boundary[i][1]], color='red')
# plt.show()


for b_value in range(len(boundary)):
    boundary[b_value][0] = round(((boundary[b_value][0] * hop_size) / float(fs)), 2)
    boundary[b_value][1] = round(((boundary[b_value][1] * hop_size + window_size) / float(fs)), 2)

# print boundary

overlapping_boundaries = [[boundary[0][0], boundary[0][1]]]
for bb in range(1, len(boundary)):
    if boundary[bb][0] >= boundary[bb-1][1]:
        overlapping_boundaries.append([boundary[bb][0], boundary[bb][1]])
    else:
        overlapping_boundaries.pop(-1)
        overlapping_boundaries.append([boundary[bb-1][0], boundary[bb][1]])
# print overlapping_boundaries
#----------------------------------------------------------------------------------------------------------------------#
"""
Enhancing the boundaries to mark vowel and non- vowel centres. Storing the result in a  text_file for .TextGrid creation.
"""
x_values = np.arange(0, len(audio_data), 1) / float(fs)

better_boundaries = [[0.00, overlapping_boundaries[0][0]]]
for bb in range(len(overlapping_boundaries) - 1):
    better_boundaries.append([overlapping_boundaries[bb][0], overlapping_boundaries[bb][1]])
    better_boundaries.append([overlapping_boundaries[bb][1], overlapping_boundaries[bb + 1][0]])
better_boundaries.append([overlapping_boundaries[-1][0], overlapping_boundaries[-1][1]])
better_boundaries.append([overlapping_boundaries[-1][1], round(x_values[-1], 2)])

print better_boundaries
#
# import matplotlib.pyplot as plt
# plt.plot(st_energy)
# plt.show()


text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Envelope\Saitama_PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid
for i in range(len(better_boundaries)):
    if i % 2 == 0:
        text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + " " + "\n")
    else:
        text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + "Vowel" + "\n")
text_file_1.close()

csvFileName = 'F:\Projects\Active Projects\Project Intern_IITB\Envelope\Saitama_PE.csv'
TGFileName = csvFileName.split('.')[0] + '_NEW.TextGrid'  # Setting name of TextGrid file

fid_csv = open(csvFileName, 'r')
fidTG = open(TGFileName, 'w')

reader = csv.reader(fid_csv, delimiter="\t")  # Reading data from csv file
data = list(reader)  # Converting read data into python list format
label_count = len(data)  # Finding total number of rows in csv file
end_time = data[-1][1]

fidTG.write('File type = "ooTextFile"\n')
fidTG.write('Object class = "TextGrid"\n')
fidTG.write('xmin = 0\n')
fidTG.write('xmax = ' + str(end_time) + '\n')
fidTG.write('tiers? <exists>\n')
fidTG.write('size = 1\n')
fidTG.write('item []:\n')
fidTG.write('\titem [1]:\n')
fidTG.write('\t\tclass = "IntervalTier"\n')
fidTG.write('\t\tname = "Labels"\n')
fidTG.write('\t\txmin = 0\n')
fidTG.write('\t\txmax = ' + str(end_time) + '\n')
fidTG.write('\t\tintervals: size = ' + str(label_count) + '\n')

for TG in range(label_count):
    fidTG.write('\t\tintervals [' + str(TG) + ']:\n')
    fidTG.write('\t\t\txmin = ' + str(data[TG][0]) + '\n')
    fidTG.write('\t\t\txmax = ' + str(data[TG][1]) + '\n')
    fidTG.write('\t\t\ttext = "' + str(data[TG][2]) + '"\n')

fid_csv.close()
fidTG.close()
#----------------------------------------------------------------------------------------------------------------------#
