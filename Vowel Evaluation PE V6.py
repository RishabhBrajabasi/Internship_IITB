from __future__ import division

import csv
import glob
import math
import os
import re
import win32api
import winsound
from datetime import datetime
from shutil import copyfile
from scipy.signal import butter, lfilter

import numpy as np
from scipy.io import wavfile

startTime = datetime.now()  # To calculate the run time of the code.

#----------------------------------------------------------------------------------------------------------------------#
# def savitzky_golay(y, window_size, order, derivation=0, rate=1):
#     import numpy as npy
#     from math import factorial
#     try:
#         window_size = npy.abs(npy.int(window_size))
#         order = npy.abs(npy.int(order))
#     except ValueError:
#         raise ValueError("window_size and order have to be of type int")
#     if window_size % 2 != 1 or window_size < 1:
#         raise TypeError("window_size size must be a positive odd number")
#     if window_size < order + 2:
#         raise TypeError("window_size is too small for the polynomials order")
#     order_range = range(order + 1)
#     half_window = (window_size - 1) // 2
#     b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
#     m = np.linalg.pinv(b).A[derivation] * rate ** derivation * factorial(derivation)
#     first_val = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
#     last_val = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
#     y = np.concatenate((first_val, y, last_val))
#     return np.convolve(m[::-1], y, mode='valid')
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
def peak_elimination(audio, window_dur=50, hop_dur=7, degree=5, threshold_smooth=100):
    new = ''
    new = new + (audio[0:76])
    new += '2\\'
    new = new + (audio[77:])
    text_file_1 = open(str(new[:-4]) + 'PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid
    try:
        audio_file = audio
        fs, data1 = wavfile.read(audio_file)  # Reading data from wav file in an array
        data1 = data1 / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
        data = butter_bandpass_filter(data1, 300, 2500, fs, order=6)
        length_o = len(data)
        window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
        hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
        window_type = np.hanning(window_size)  # Window type: Hanning (by default)
        no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
        zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
        data = np.concatenate((data, zero_array))
        length = len(data)  # Finding length of the actual data
    #----------------------------------------------------------------------------------------------------------------------#
        noise_energy = 0  # Initializing noise energy
        energy = [0] * length  # Initializing list energy
        for bit in range(length):
            energy[bit] = data[bit] * data[bit]  # Squaring each point of the data to calculate noise energy
        for ne in range(0, 800):
            noise_energy += energy[ne]  # Taking the first 800 samples of the original sound file
        noise_energy /= 800  # Averaging the square of the first 800 noise samples
    #----------------------------------------------------------------------------------------------------------------------#
        st_energy = []
        for i in range(no_frames):  # Calculating frame wise short term energy
            frame = data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
            st_energy.append(sum(frame ** 2))  # Calculating the short term energy
        max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
        for i in range(no_frames):
            st_energy[i] = st_energy[i]/max_st_energy  # Normalizing the curve
    #----------------------------------------------------------------------------------------------------------------------#
        if len(st_energy) < threshold_smooth:
            st_energy = st_energy
        else:
            st_energy = moving_average(st_energy, 20)
    #----------------------------------------------------------------------------------------------------------------------#
        peak = []  # Initializing list
        count_of_peaks = 0  # Initializing no of peaks
        for p in range(len(st_energy)):
            if p == 0:  # First element
                if st_energy[p] > st_energy[p + 1]:  # If the first element is greater than the succeeding element it is a peak.
                    peak.append(st_energy[p])  # Append the energy level of the peak
                    count_of_peaks += 1  # Increment count
                else:
                    peak.append(0)  # Else append a zero
            elif p == len(st_energy) - 1:  # Last element
                if st_energy[p] > st_energy[p - 1]:  # If the last element is greater than the preceding element it is a peak.
                    peak.append(st_energy[p])  # Append the energy level of the peak
                    count_of_peaks += 1  # Increment count
                else:
                    peak.append(0)  # Else append a zero
            else:  # All the other elements
                if st_energy[p] > st_energy[p + 1] and st_energy[p] > st_energy[p - 1]:  # If the element is greater than the element preceding and succeeding it, it is a peak.
                    peak.append(st_energy[p])  # Append the energy level of the peak
                    count_of_peaks += 1  # Increment count
                else:
                    peak.append(0)  # Else append a zero
    #----------------------------------------------------------------------------------------------------------------------#
        threshold = 0.01 + 0.04 * (noise_energy + (sum(peak) / count_of_peaks))  # The threshold which eliminates minor peaks.
    #----------------------------------------------------------------------------------------------------------------------#
        count_of_peaks_threshold = 0
        peak_threshold = []
        location_peak = []
        for p in range(len(peak)):
            if threshold < peak[p]:  # If the peak value is greater than the threshold
                peak_threshold.append(peak[p])  # Append the energy level to a new list
                count_of_peaks_threshold += 1  # Increment count
                location_peak.append(p)  # Make note of the location of the peak
            else:
                peak_threshold.append(0)  # Else append zero
    #----------------------------------------------------------------------------------------------------------------------#
        valley = []
        count_of_valleys = 0
        location_valley = []
        for p in range(len(st_energy)):
            if p == 0:  # For the first element
                if st_energy[p] < st_energy[p + 1]:  # If the first element is lesser than the succeeding element
                    valley.append(st_energy[p])  # Append the energy level of the valley
                    count_of_valleys += 1  # Increment the count
                    location_valley.append(p)  # Make note of the position of the valley
                else:
                    valley.append(0)  # Else append zero
            elif p == len(st_energy) - 1:  # For the last element
                if st_energy[p] < st_energy[p - 1]:  # If the last element is lesser than the preceding element
                    valley.append(st_energy[p])  # Append the energy level of the valley
                    count_of_valleys += 1  # Increment the count
                    location_valley.append(p)  # Make note of the position of the valley
                else:
                    valley.append(0)  # Else append zero
            else:
                if st_energy[p] < st_energy[p + 1] and st_energy[p] < st_energy[
                            p - 1]:  # If the element is lesser than the element preceding and succeeding it
                    valley.append(st_energy[p])  # Append the energy level of the valley
                    count_of_valleys += 1  # Increment the count
                    location_valley.append(p)  # Make note of the position of the valley
                else:
                    valley.append(0)  # Else append zero
    #----------------------------------------------------------------------------------------------------------------------#
        location = location_peak + location_valley  # Combing the list of the location of all the peaks and valleys
        location.sort()  # Sorting it so that each peak has a valley to it's left and right
        ripple_valley = []
        ripple_peak = []
        ripple = []
        # What we need is only the valleys to the left and right of the peak. The other valleys are not important
        for k in range(len(location_peak)):
            q = location.index(location_peak[k])  # Extracting the location of the peak
            if location_peak[k] == len(peak) - 1:  # If the peak is the last element of the short term energy curve
                ripple.append(location[q - 1])  # The location of the valley before the last peak is added
                ripple_valley.append(location[q - 1])  # The location of the valley before the last peak is added
                ripple.append(location[q])  # The location of the peak is added
                ripple_peak.append(location[q])  # The location of the peak is added
                ripple.append(location[q - 1])  # The location of the valley before the last peak is added, as there is no valley after it
                ripple_valley.append(location[q - 1])  # The location of the valley before the last peak is added, as there is no valley after it
            elif location_peak[k] == 0:  # If the peak is the first element of the short term energy curve
                ripple.append(location[q + 1])  # The location of the valley after the first peak is added
                ripple_valley.append(location[q + 1])  # The location of the valley after the first peak is added
                ripple.append(location[q])  # The location of the peak is added
                ripple_peak.append(location[q])  # The location of the peak is added
                ripple.append(location[q + 1])  # The location of the valley after the first peak is added, as there is no valley after it
                ripple_valley.append(location[q + 1])  # The location of the valley after the first peak is added, as there is no valley after it
            else:  # For every other element
                ripple.append(location[q - 1])  # The location of the valley before the peak is added
                ripple_valley.append(location[q - 1])  # The location of the valley before the peak is added
                ripple.append(location[q])  # The location of the peak is added
                ripple_peak.append(location[q])  # The location of the peak is added
                ripple.append(location[q + 1])  # The location of the valley after the peak is added
                ripple_valley.append(location[q + 1])  # The location of the valley after the peak is added
    #----------------------------------------------------------------------------------------------------------------------#
        ripple_value = []
        for k in range(1, len(ripple), 3):
            ripple_value.append((st_energy[ripple[k]] - st_energy[ripple[k + 1]]) / (st_energy[ripple[k]] - st_energy[ripple[k - 1]]))
    #----------------------------------------------------------------------------------------------------------------------#
        ripple_value_thresh = []
        count_of_vowels = 0
        ripple_upper_limit = 2.0
        ripple_lower_limit = 0.4
        for k in range(len(ripple_value)):
            if k != len(ripple_value) - 1:
                if ripple_upper_limit >= ripple_value[k] >= ripple_lower_limit:
                    ripple_value_thresh.append(ripple_value[k])
                    count_of_vowels += 1
                elif ripple_value[k] >= ripple_upper_limit and ripple_value[k + 1] <= ripple_lower_limit and \
                                        ripple_peak[ripple_value.index(ripple_value[k + 1])] - ripple_peak[
                                    ripple_value.index(ripple_value[k])] < 20:
                    ripple_value_thresh.append(ripple_value[k])
                    count_of_vowels += 1
                elif ripple_value[k] <= ripple_lower_limit and ripple_value[k + 1] >= ripple_upper_limit and \
                                        ripple_peak[ripple_value.index(ripple_value[k + 1])] - ripple_peak[
                                    ripple_value.index(ripple_value[k])] < 20:
                    ripple_value_thresh.append(ripple_value[k + 1])
                    count_of_vowels += 1
                else:
                    ripple_value_thresh.append(0)
            else:
                if ripple_upper_limit >= ripple_value[k] >= ripple_lower_limit:
                    ripple_value_thresh.append(ripple_value[k])
                    count_of_vowels += 1
                else:
                    ripple_value_thresh.append(0)

        loc = []
        peak_threshold[:] = []
        for p in range(len(ripple_value_thresh)):  # Finding the location of the peaks that have been chosen
            if ripple_value_thresh[p] != 0:
                loc.append(location_peak[ripple_value.index(ripple_value_thresh[p])])

        for p in range(no_frames):  # Finding the energy level of the peaks that have been chosen
            if p in loc:
                peak_threshold.append(st_energy[loc.index(p)])
            else:
                peak_threshold.append(0)
    #----------------------------------------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------------------------------------#
        mark = []
        for p in range(len(peak_threshold)):  # Extracting a 50 ms slice of the audio file based on the frame number
            if peak_threshold[p] is not 0:
                mark.append(p * hop_size)
                mark.append(p * hop_size + window_size)

        if mark[0] != 0.0:
            text_file_1.write('%06.3f' % (0 * 0.0000625) + "\t" + '%06.3f' % (mark[0] * 0.0000625) + "\t" + " " + "\n")

        for i in range(0, len(mark)-1):  # Writing the result to a CSV File
            if i%2 == 0:
                text_file_1.write(
                '%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + "Vowel" + "\n")
            else:
                text_file_1.write(
                '%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + " " + "\n")


        if mark[-1] != length_o/16000 - 0.02:
            text_file_1.write('%06.3f' % (mark[-1] * 0.0000625) + "\t" + '%06.3f' % (length_o/16000 - 0.02) + "\t" + " " + "\n")

        text_file_1.close()

        return new, len(loc)
    except:
        return new, -1
#----------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------#
def fa_count(text_file):
    text_grid_1 = open(text_file, 'r')  # Open TextGrid in read mode
    data_1 = text_grid_1.read()  # Read the data from the TextGrid
    try:
        def count(vowel):  # Function to count the number of times the argument 'vowel' occurs in the TextGrid
            return data_1.count(vowel)
        c1 = count('"aa"')  # For vowel sound "aa"
        c2 = count('"AA"')  # For vowel sound "AA"
        c3 = count('"ae"')  # For vowel sound "ae"
        c4 = count('"aw"')  # For vowel sound "aw"
        c5 = count('"ay"')  # For vowel sound "ay"
        c6 = count('"ee"')  # For vowel sound "ee"
        c7 = count('"ex"')  # For vowel sound "ex"
        c8 = count('"ii"')  # For vowel sound "ii"
        c9 = count('"II"')  # For vowel sound "II"
        c10 = count('"oo"')  # For vowel sound "oo"
        c11 = count('"OO"')  # For vowel sound "OO"
        c12 = count('"oy"')  # For vowel sound "oy"
        c13 = count('"uu"')  # For vowel sound "uu"
        c14 = count('"UU"')  # For vowel sound "UU"
        c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14  # Summing up all the vowel occurrences
        start = data_1.find('xmin')  # The first occurrence of 'xmin', points to the staring time of the Audio File
        stop = data_1.find('xmax')  # The first occurrence of 'xmax', points to the ending time of the Audio File
        starting = float(data_1[start + 7])  # Extracting the starting time.(Always zero)
        stopping = float(data_1[stop + 7] + data_1[stop + 8] + data_1[stop + 9] + data_1[stop + 10] + data_1[stop + 11] + data_1[stop + 12])  # Extracting the stopping time.
        duration = stopping - starting  # Calculating the duration of the Audio File
        return c, duration
    except:
        return -1, -1
#----------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------#
def textgrid(csvi):
    try:
        csvFileName = csvi
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
    except:
        print 'Text Grid Creation Error', csvi
#----------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------#
def evaluation(textgrid1, textgrid2):
    try:
        text_grid_1 = open(textgrid1, 'r')  # Open the FA TextGrid
        text_grid_2 = open(textgrid2, 'r')  # Open the TextGrid created by the script
        data_1 = text_grid_1.read()  # Read and assign the content of the FA TextGrid to data_1
        data_2 = text_grid_2.read()  # Read and assign the content of the created TextGrid to data_2
        time_1 = []  # Creating an empty list to record time
        time_2 = []
        counter = 0
#----------------------------------------------------------------------------------------------------------------------#
        for m in re.finditer('text = "', data_1):
            if data_1[m.start() - 33] == '=':
                time_1.append(float(
                    data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] + data_1[m.start() - 29] +
                    data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
                time_1.append(float(
                    data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[m.start() - 10] +
                    data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] +
                    data_1[m.start() - 5]))
            else:
                time_1.append(float(
                    data_1[m.start() - 33] + data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] +
                    data_1[m.start() - 29] + data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
                time_1.append(float(
                    data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[m.start() - 10] +
                    data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] +
                    data_1[m.start() - 5]))
#----------------------------------------------------------------------------------------------------------------------#
            if data_1[m.start() + 9] == '"':
                time_1.append(data_1[m.start() + 8])
            elif data_1[m.start() + 10] == '"':
                time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9])
            else:
                time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9] + data_1[m.start() + 10])

            time_1.append(counter)
            counter += 1
#----------------------------------------------------------------------------------------------------------------------#
        for m in re.finditer('"Vowel"', data_2):
            time_2.append(float(
                data_2[m.start() - 34] + data_2[m.start() - 33] + data_2[m.start() - 32] + data_2[m.start() - 31] +
                data_2[m.start() - 30] + data_2[m.start() - 29]))
            time_2.append(float(
                data_2[m.start() - 17] + data_2[m.start() - 16] + data_2[m.start() - 15] + data_2[m.start() - 14] +
                data_2[m.start() - 13] + data_2[m.start() - 12]))
#----------------------------------------------------------------------------------------------------------------------#
        listing = []

        for outer in range(0, len(time_2), 2):
            for inner in range(0, len(time_1), 4):
                if time_1[inner] <= time_2[outer] < time_1[inner + 1] and time_1[inner] < time_2[outer + 1] <= time_1[
                            inner + 1]:
                    listing.append(time_1[inner + 2])
                    listing.append(time_1[inner + 3])

        for outer in range(0, len(time_2), 2):
            for inner in range(0, len(time_1) - 4, 4):
                if time_1[inner] < time_2[outer] < time_1[inner + 1] and time_1[inner + 4] < time_2[outer + 1] < time_1[
                            inner + 5]:
                    listing.append(time_1[inner + 2])
                    listing.append(time_1[inner + 3])
                    listing.append(time_1[inner + 6])
                    listing.append(time_1[inner + 7])

        count = 0
        vowel_data = ['aa', 'AA', 'ae', 'aw', 'ay', 'ee', 'ex', 'ii', 'II', 'oo', 'OO', 'oy', 'uu', 'UU']

        already_here = []
        for vowel_sound in range(0, len(listing), 2):
            if listing[vowel_sound] in vowel_data and listing[vowel_sound + 1] not in already_here:
                count += 1
                already_here.append(listing[vowel_sound + 1])

        return count
    except:
        return -1
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
def results_write(sr_no, name, vpe, vpee, vfa, t):
    if vpe == 0 or vfa == 0:
        results_vowel.write(str(sr_no) + ',')
        results_vowel.write(str(name) + ',')
        results_vowel.write(str(vpe) + ',')
        results_vowel.write(str(vpee) + ',')
        results_vowel.write(str(vfa) + ',')
        results_vowel.write(str(t) + ',')
        results_vowel.write(str(0) + ',')
        results_vowel.write(str(0) + ',')
        results_vowel.write('Error' + ',')
        results_vowel.write('\n')

    else:
        results_vowel.write(str(sr_no) + ',')
        results_vowel.write(str(name) + ',')
        results_vowel.write(str(vpe) + ',')
        results_vowel.write(str(vpee) + ',')
        results_vowel.write(str(vfa) + ',')
        results_vowel.write(str(t) + ',')
        results_vowel.write(str(vpee / float(vpe)) + ',')
        results_vowel.write(str(vpee / float(vfa)) + ',')
        results_vowel.write('Fine' + ',')
        results_vowel.write('\n')
#----------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------#
file_name_template_1 = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\\'
result_name = 'Vowel_Evaluation_V6_Test_7_M2.csv'

copyfile('C:\Users\Mahe\PycharmProjects\untitled\IIT-B Internship\\Vowel Evaluation PE V6.py',
         'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\\' + result_name[:-4] + '.txt')

if not os.path.exists(file_name_template_1 + "Analyze\\" + result_name[:-4]):
    os.makedirs(file_name_template_1 + "Analyze\\" + result_name[:-4])

#----------------------------------------------------------------------------------------------------------------------#
file_no = 1
results_vowel = open(file_name_template_1 + result_name, 'w')  # The csv file where the results are saved
only_audio = glob.glob(file_name_template_1 + 'Data 1\*.wav')  # Extract file name of all audio samples
only_text = glob.glob(file_name_template_1 + '\Data 1\*.TextGrid')  # Extract file name of all force aligned TextGrid
results_vowel.write(
    'File No' + ',' + 'Name' + ',' + 'Peak_Elimination' + ',' + 'Evaluation Script PE' + ',' + 'Vowel Count FA' +
    ',' + 'Duration(In Seconds)' + ',' + 'Precision PE' + ',' + 'Recall PE' + ',' + 'Tag' + '\n')  # Column headers
#----------------------------------------------------------------------------------------------------------------------#

for j in only_audio:
    filename1, count1 = peak_elimination(j, window_dur=50, hop_dur=7,degree=7, threshold_smooth=120)  # Run Peak elimination algorithm
    textgrid(str(filename1[:-4] + 'PE.csv'))  # Create TextGrid based on results of peak elimination
    count_pe = evaluation(str(j[:-4] + '.TextGrid'), str(filename1[:-4] + 'PE_NEW.TextGrid'))  # Evaluate the TextGrid created and the force aligned TextGrid
    vowel_count, time = fa_count(str(j[:-4] + '.TextGrid'))  # Count the number of vowels in the audio file according to the force aligned TextGrid
    results_write(file_no, j, count1, count_pe, vowel_count, time)  # Writing the results in the csv file.
    # ----------------------------------------------------------------------------------------------------------------------#
    copyfile(j, file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + '.wav')
    copyfile(str(j[:-4] + '.TextGrid'),file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'FA.TextGrid')
    copyfile(str(filename1[:-4] + 'PE_NEW.TExtGrid'),file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'PE.TextGrid')
    file_no += 1

print datetime.now() - startTime  # Print program run time
winsound.Beep(300, 2000)
win32api.MessageBox(0, 'The code has finished running', 'Complete', 0x00001000)
