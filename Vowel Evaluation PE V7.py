"""
Vowel Evaluation PE V7.py
Test different features and ideas.
"""
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
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#
def peaks(st_energy_peak):
    peak_f = []  # Initializing list
    count_of_peaks_f = 0  # Initializing no of peaks
    for p in range(len(st_energy_peak)):
        if p == 0:  # First element
            peak_f.append(0)  # Append the energy level of the peak
        elif p == len(st_energy_peak) - 1:  # Last element
            peak_f.append(0)  # Append the energy level of the peak
        else:  # All the other elements
            if st_energy_peak[p] > st_energy_peak[p + 1] and st_energy_peak[p] > st_energy_peak[p - 1]:  # If the element is greater than the element preceding and succeeding it, it is a peak.
                peak_f.append(st_energy_peak[p])  # Append the energy level of the peak
                count_of_peaks_f += 1  # Increment count
            else:
                peak_f.append(0)  # Else append a zero
    return count_of_peaks_f, peak_f
#----------------------------------------------------------------------------------------------------------------------#
def threshold_peak(peak, threshold):
    peak_f = []
    location_peak_f = []
    for p in range(len(peak)):
        if threshold < peak[p]:  # If the peak value is greater than the threshold
            peak_f.append(peak[p])  # Append the energy level to a new list
            location_peak_f.append(p)  # Make note of the location of the peak
        else:
            peak_f.append(0)  # Else append zero
    return peak_f, location_peak_f
#----------------------------------------------------------------------------------------------------------------------#
def valleys(st_energy_valley):
    valley_f = []
    location_valley_f = []
    for v in range(len(st_energy_valley)):
        if v == 0:  # For the first element
            if st_energy_valley[v] < st_energy_valley[v + 1]:  # If the first element is lesser than the succeeding element
                valley_f.append(st_energy_valley[v])  # Append the energy level of the valley
                location_valley_f.append(v)  # Make note of the position of the valley
            else:
                valley_f.append(0)  # Else append zero
        elif v == len(st_energy_valley) - 1:  # For the last element
            if st_energy_valley[v] < st_energy_valley[v - 1]:  # If the last element is lesser than the preceding element
                valley_f.append(st_energy_valley[v])  # Append the energy level of the valley
                location_valley_f.append(v)  # Make note of the position of the valley
            else:
                valley_f.append(0)  # Else append zero
        else:
            if st_energy_valley[v] < st_energy_valley[v + 1] and st_energy_valley[v] < st_energy_valley[v - 1]:  # If the element is lesser than the element preceding and succeeding it
                valley_f.append(st_energy_valley[v])  # Append the energy level of the valley
                location_valley_f.append(v)  # Make note of the position of the valley
            else:
                valley_f.append(0)  # Else append zero
    return valley_f, location_valley_f
#----------------------------------------------------------------------------------------------------------------------#
def ripple_calculation(st_energy_ripple, peak_ripple, location_ripple, location_peak_ripple, location_valley_ripple):
    ripple_f = []
    for k in range(len(location_peak_ripple)):
        q = location_ripple.index(location_peak_ripple[k])  # Extracting the location of the peak
        if location_peak_ripple[k] == len(peak_ripple) - 1:  # If the peak is the last element of the short term energy curve
            ripple_f.append(location_ripple[q - 1])  # The location of the valley before the last peak is added
            ripple_f.append(location_ripple[q])  # The location of the peak is added
            ripple_f.append(location_ripple[q - 1])  # The location of the valley before the last peak is added, as there is no valley after it
        elif location_peak_ripple[k] == 0:  # If the peak is the first element of the short term energy curve
            ripple_f.append(location_ripple[q + 1])  # The location of the valley after the first peak is added
            ripple_f.append(location_ripple[q])  # The location of the peak is added
            ripple_f.append(location_ripple[q + 1])  # The location of the valley after the first peak is added, as there is no valley after it
        else:  # For every other element
            ripple_f.append(location_ripple[q - 1])  # The location of the valley before the peak is added
            ripple_f.append(location_ripple[q])  # The location of the peak is added
            ripple_f.append(location_ripple[q + 1])  # The location of the valley after the peak is added
    ripple_value_f = []
    for k in range(1, len(ripple_f), 3):
        ripple_value_f.append((st_energy_ripple[ripple_f[k]] - st_energy_ripple[ripple_f[k + 1]]) / (st_energy_ripple[ripple_f[k]] - st_energy_ripple[ripple_f[k - 1]]))
    return ripple_value_f
#----------------------------------------------------------------------------------------------------------------------#
def ripple_elimination(ripple_value_elimination, location_peak_elimination, st_energy_elimination):
    loc = []
    for k in range(len(ripple_value_elimination)):
        loc.append(location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k])])

    for k in range(len(ripple_value_elimination)):
        if k != len(ripple_value_elimination) - 1:
            if location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k + 1])] - location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k])] < 20:
                if ripple_value_elimination[k] > 3.0 and ripple_value_elimination[k + 1] < 1.4 or ripple_value_elimination[k] > 1.02 and ripple_value_elimination[
                            k + 1] < 0.3:
                    v1 = st_energy_elimination[location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k])]]
                    v2 = st_energy_elimination[location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k + 1])]]
                    if v1 >= v2:
                        loc.remove(location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k + 1])])
                    else:
                        loc.remove(location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k])])
        else:
            if ripple_value_elimination[k] > 3.0:
                loc.remove(location_peak_elimination[ripple_value_elimination.index(ripple_value_elimination[k])])

    boundary_f = []
    for j in range(len(st_energy_elimination)):
        if j in loc:
            boundary_f.append(st_energy_elimination[loc.index(j)])
        else:
            boundary_f.append(0)
    return boundary_f
#----------------------------------------------------------------------------------------------------------------------#
def peak_elimination(audio, window_dur=50, hop_dur=7, degree=5, threshold_smooth=100):
    new = ''
    start = audio.find('Data 1')
    new = new + (audio[0:start + 5])
    new += '2\\'
    new = new + (audio[77:])
    text_file_1 = open(str(new[:-4]) + 'PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid
    try:
        audio_file = audio
        fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
        data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
        data = butter_bandpass_filter(data, 300, 2500, fs, order=6)  # Filter to remove fricatives
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
        count_of_peaks, peak = peaks(st_energy)
    #----------------------------------------------------------------------------------------------------------------------#
        threshold = 0.01 + 0.04 * (noise_energy + (sum(peak) / count_of_peaks))  # The threshold which eliminates minor peaks.
    #----------------------------------------------------------------------------------------------------------------------#
        peak_threshold, location_peak = threshold_peak(peak, threshold)
    #----------------------------------------------------------------------------------------------------------------------#
        valley, location_valley = valleys(st_energy)
    #----------------------------------------------------------------------------------------------------------------------#
        location = location_peak + location_valley  # Combing the list of the location of all the peaks and valleys
        location.sort()  # Sorting it so that each peak has a valley to it's left and right
        ripple_value = ripple_calculation(st_energy, peak, location, location_peak, location_valley)
    #----------------------------------------------------------------------------------------------------------------------#
        boundary = ripple_elimination(ripple_value, location_peak, st_energy)
    #----------------------------------------------------------------------------------------------------------------------#
        mark = []
        vowel = 0
        for p in range(len(boundary)):  # Extracting a 50 ms slice of the audio file based on the frame number
            if boundary[p] is not 0:
                mark.append(p * hop_size)
                mark.append(p * hop_size + window_size)
                vowel += 1

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

        return new, vowel
    except:
        return new, -1
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
        results_vowel.write(str(-1) + ',')
        results_vowel.write(str(-1) + ',')
        results_vowel.write(str(-1) + ',')
        results_vowel.write(str(-1) + ',')
        results_vowel.write(str(-1) + ',')
        results_vowel.write(str(-1) + ',')
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
        sr1 = vpe/t
        sr2 = vpee/t
        sr3 = vfa/t
        results_vowel.write(str(sr1) + ',')
        results_vowel.write(str(sr2) + ',')
        results_vowel.write(str(sr3) + ',')
        if 0.0 <= sr1 < 3.0:
            results_vowel.write('Slow' + ',')
        elif 3.0 <= sr1 < 5.0:
            results_vowel.write('Normal' + ',')
        else:
            results_vowel.write('Fast' + ',')

        if 0.0 <= sr2 < 3.0:
            results_vowel.write('Slow' + ',')
        elif 3.0 <= sr2 < 5.0:
            results_vowel.write('Normal' + ',')
        else:
            results_vowel.write('Fast' + ',')

        if 0.0 <= sr3 < 3.0:
            results_vowel.write('Slow' + ',')
        elif 3.0 <= sr3 < 5.0:
            results_vowel.write('Normal' + ',')
        else:
            results_vowel.write('Fast' + ',')
        results_vowel.write('\n')
#----------------------------------------------------------------------------------------------------------------------#
file_name_template_1 = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V7\\'
result_name = 'Vowel_Evaluation_V7_Modularization.csv'
copyfile('C:\Users\Mahe\PycharmProjects\Internship_IITB\\Vowel Evaluation PE V7.py',
         'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V7\\' + result_name[:-4] + '.txt')

if not os.path.exists(file_name_template_1 + "Analyze\\" + result_name[:-4]):
    os.makedirs(file_name_template_1 + "Analyze\\" + result_name[:-4])
#----------------------------------------------------------------------------------------------------------------------#
file_no = 1
results_vowel = open(file_name_template_1 + result_name, 'w')  # The csv file where the results are saved
only_audio = glob.glob(file_name_template_1 + 'Data 1\*.wav')  # Extract file name of all audio samples
only_text = glob.glob(file_name_template_1 + '\Data 1\*.TextGrid')  # Extract file name of all force aligned TextGrid
results_vowel.write(
    'File No' + ',' + 'Name' + ',' + 'Peak_Elimination' + ',' + 'Evaluation Script PE' + ',' + 'Vowel Count FA' +
    ',' + 'Duration(In Seconds)' + ',' + 'Precision PE' + ',' + 'Recall PE' + ',' + 'Tag' + ',' + 'Speaking rate 1' + ',' +
    'Speaking rate 2' + ',' + 'Speaking rate 3' + ',' + 'Speed 1' + ',' + 'Speed 2' + ',' + 'Speed 3' + '\n')  # Column headers
#----------------------------------------------------------------------------------------------------------------------#

for j in only_audio:
    filename1, count1 = peak_elimination(j, window_dur=50, hop_dur=7, degree=7, threshold_smooth=120)  # Run Peak elimination algorithm
    textgrid(str(filename1[:-4] + 'PE.csv'))  # Create TextGrid based on results of peak elimination
    count_pe = evaluation(str(j[:-4] + '.TextGrid'), str(filename1[:-4] + 'PE_NEW.TextGrid'))  # Evaluate the TextGrid created and the force aligned TextGrid
    vowel_count, time = fa_count(str(j[:-4] + '.TextGrid'))  # Count the number of vowels in the audio file according to the force aligned TextGrid
    results_write(file_no, j, count1, count_pe, vowel_count, time)  # Writing the results in the csv file.
    # ----------------------------------------------------------------------------------------------------------------------#
    copyfile(j, file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + '.wav')
    copyfile(str(j[:-4] + '.TextGrid'), file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'FA.TextGrid')
    copyfile(str(filename1[:-4] + 'PE_NEW.TExtGrid'), file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'PE.TextGrid')
    file_no += 1

print datetime.now() - startTime  # Print program run time
winsound.Beep(300, 2000)
win32api.MessageBox(0, 'The code has finished running', 'Complete', 0x00001000)

results_vowel.close()

results_vowel = open(file_name_template_1 + result_name, 'r')  # The csv file where the results are saved
file_name_template_2 = file_name_template_1 + result_name[:-4] + '_Analysis.csv'
results_analysis = open(file_name_template_2, 'w')  # The csv file where the results are saved
data = results_vowel.read()
one = data.split('\n')
two = []
for i in range(len(one)):
    two.append(one[i].split(','))
two.pop(0)
two.pop(-1)
results_analysis.write('Start time' + ',' + 'End time' + ',' + 'Count' + ',' + 'Precision' + ',' + 'Recall' + ',' + 'Files' + '\n')

def results(analyze, time_1, time_2):
    count = 0
    precision = []
    recall = []
    names = []
    for element in range(len(one) - 2):
        if analyze[element][8] == 'Fine':
            if time_1 < float(analyze[element][5]) <= time_2:
                    count += 1
                    precision.append(float(analyze[element][6]))
                    recall.append(float(analyze[element][7]))
                    names.append(analyze[element][0])
    results_analysis.write(str(start_time) + ',' + str(end_time) + ',' + str(count) + ',' + str(sum(precision)/len(precision)) + ','
                           + str(sum(recall)/len(recall)) + ',' + str(names) + ',' + '\n')

start_time = 0.0
end_time = 0.5
results(two, start_time, end_time)

start_time = 0.5
end_time = 1.0
results(two, start_time, end_time)

start_time = 1.0
end_time = 1.5
results(two, start_time, end_time)

start_time = 1.5
end_time = 2.0
results(two, start_time, end_time)

start_time = 2.0
end_time = 2.5
results(two, start_time, end_time)

start_time = 2.5
end_time = 3.0
results(two, start_time, end_time)

start_time = 3.0
end_time = 3.5
results(two, start_time, end_time)

start_time = 3.5
end_time = 4.0
results(two, start_time, end_time)

start_time = 4.0
end_time = 4.5
results(two, start_time, end_time)

start_time = 4.5
end_time = 5.0
results(two, start_time, end_time)

start_time = 5.0
end_time = 5.5
results(two, start_time, end_time)

start_time = 5.5
end_time = 6.0
results(two, start_time, end_time)

start_time = 6.0
end_time = 6.5
results(two, start_time, end_time)

start_time = 6.5
end_time = 7.0
results(two, start_time, end_time)

start_time = 7.0
end_time = 7.5
results(two, start_time, end_time)

start_time = 7.5
end_time = 8.0
results(two, start_time, end_time)

start_time = 8.0
end_time = 8.5
results(two, start_time, end_time)

start_time = 8.5
end_time = 9.0
results(two, start_time, end_time)

start_time = 9.0
end_time = 9.5
results(two, start_time, end_time)

start_time = 9.5
end_time = 10.0
results(two, start_time, end_time)

start_time = 10.0
end_time = 10.5
results(two, start_time, end_time)

start_time = 10.5
end_time = 11.0
results(two, start_time, end_time)

start_time = 11.0
end_time = 11.5
results(two, start_time, end_time)

start_time = 11.5
end_time = 12.0
results(two, start_time, end_time)

start_time = 12.0
end_time = 60.0
results(two, start_time, end_time)
