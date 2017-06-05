from __future__ import division

import csv
import glob
import os
import re
import sys
import math
import win32api
import winsound
from datetime import datetime
from shutil import copyfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

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
#----------------------------------------------------------------------------------------------------------------------#


def peak_elimination(audio):
    new = ''
    start = audio.find('Data 1')
    new = new + (audio[0:start + 5])
    new += '2\\'
    new = new + (audio[start+8:])
    text_file_1 = open(str(new[:-4]) + 'PE.csv', 'w')
    try:
        audio_file = audio
        window_dur = 50
        hop_dur = 7
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

        for b_value in range(len(boundary)):
            boundary[b_value][0] = round(((boundary[b_value][0] * hop_size) / float(fs)), 2)
            boundary[b_value][1] = round(((boundary[b_value][1] * hop_size + window_size) / float(fs)), 2)

        overlapping_boundaries = [[boundary[0][0], boundary[0][1]]]
        for bb in range(1, len(boundary)):
            if boundary[bb][0] > boundary[bb-1][1]:
                overlapping_boundaries.append([boundary[bb][0], boundary[bb][1]])
            else:
                overlapping_boundaries.pop(-1)
                overlapping_boundaries.append([boundary[bb-1][0], boundary[bb][1]])
        #----------------------------------------------------------------------------------------------------------------------#
        """
        Enhancing the boundaries to mark vowel and non- vowel centres. Storing the result in a  text_file for .TextGrid creation.
        """
        x_values = np.arange(0, len(audio_data), 1) / float(fs)

        if overlapping_boundaries[0][0] == 0:
            better_boundaries = [[]]
            for bb in range(len(overlapping_boundaries) - 1):
                better_boundaries.append([overlapping_boundaries[bb][0], overlapping_boundaries[bb][1]])
                better_boundaries.append([overlapping_boundaries[bb][1], overlapping_boundaries[bb + 1][0]])
            better_boundaries.append([overlapping_boundaries[-1][0], overlapping_boundaries[-1][1]])
            better_boundaries.append([overlapping_boundaries[-1][1], round(x_values[-1], 2)])
            better_boundaries.pop(0)
        else:
            better_boundaries = [[0.00, overlapping_boundaries[0][0]]]
            for bb in range(len(overlapping_boundaries) - 1):
                better_boundaries.append([overlapping_boundaries[bb][0], overlapping_boundaries[bb][1]])
                better_boundaries.append([overlapping_boundaries[bb][1], overlapping_boundaries[bb + 1][0]])
            better_boundaries.append([overlapping_boundaries[-1][0], overlapping_boundaries[-1][1]])
            better_boundaries.append([overlapping_boundaries[-1][1], round(x_values[-1], 2)])

        for i in range(len(better_boundaries)):
            if better_boundaries[0][0] != overlapping_boundaries[0][0] and better_boundaries[0][1] != overlapping_boundaries[0][1]:
                if i % 2 == 0:
                    text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + " " + "\n")
                else:
                    text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + "Vowel" + "\n")
            else:
                if i % 2 == 0:
                    text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + "Vowel" + "\n")
                else:
                    text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + " " + "\n")
        text_file_1.close()
        return new, len(overlapping_boundaries)
    except:
        return new, -1

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
        stopping = float(
            data_1[stop + 7] + data_1[stop + 8] + data_1[stop + 9] + data_1[stop + 10] + data_1[stop + 11] + data_1[
                stop + 12])  # Extracting the stopping time.
        duration = stopping - starting  # Calculating the duration of the Audio File
        return c, duration
    except:
        return -1, -1

# ----------------------------------------------------------------------------------------------------------------------#
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

# ----------------------------------------------------------------------------------------------------------------------#
def evaluation(textgrid1, textgrid2):
    try:
        text_grid_1 = open(textgrid1, 'r')  # Open the FA TextGrid
        text_grid_2 = open(textgrid2, 'r')  # Open the TextGrid created by the script
        data_1 = text_grid_1.read()  # Read and assign the content of the FA TextGrid to data_1
        data_2 = text_grid_2.read()  # Read and assign the content of the created TextGrid to data_2
        time_1 = []  # Creating an empty list to record time
        time_2 = []
        counter = 0
        # ----------------------------------------------------------------------------------------------------------------------#

        for m in re.finditer('text = "', data_1):
            if data_1[m.start() - 33] == '=':
                time_1.append(float(
                    data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] + data_1[
                        m.start() - 29] +
                    data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
                time_1.append(float(
                    data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[
                        m.start() - 10] +
                    data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] +
                    data_1[m.start() - 5]))
            else:
                time_1.append(float(
                    data_1[m.start() - 33] + data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[
                        m.start() - 30] +
                    data_1[m.start() - 29] + data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[
                        m.start() - 26]))
                time_1.append(float(
                    data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[
                        m.start() - 10] +
                    data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] +
                    data_1[m.start() - 5]))
                # ----------------------------------------------------------------------------------------------------------------------#
            if data_1[m.start() + 9] == '"':
                time_1.append(data_1[m.start() + 8])
            elif data_1[m.start() + 10] == '"':
                time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9])
            else:
                time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9] + data_1[m.start() + 10])

            time_1.append(counter)
            counter += 1
            # ----------------------------------------------------------------------------------------------------------------------#

        counter_vowel = 0
        for m in re.finditer('"Vowel"', data_2):
            time_2.append(float(
                data_2[m.start() - 34] + data_2[m.start() - 33] + data_2[m.start() - 32] + data_2[m.start() - 31] +
                data_2[m.start() - 30] + data_2[m.start() - 29]))
            time_2.append(float(
                data_2[m.start() - 17] + data_2[m.start() - 16] + data_2[m.start() - 15] + data_2[m.start() - 14] +
                data_2[m.start() - 13] + data_2[m.start() - 12]))
            time_2.append(counter_vowel)
            counter_vowel += 1
            # ----------------------------------------------------------------------------------------------------------------------#

        vowel_data = ['aa', 'AA', 'ae', 'aw', 'ay', 'ee', 'ex', 'ii', 'II', 'oo', 'OO', 'oy', 'uu', 'UU']

        fa_vowel_time = [[]]
        for tick in range(0, len(time_1), 4):
            if time_1[tick + 2] in vowel_data:
                fa_vowel_time.append([time_1[tick], time_1[tick + 1], time_1[tick + 2], time_1[tick + 3]])
        fa_vowel_time.pop(0)

        ev_vowel_time = [[]]
        for tock in range(0, len(time_2), 3):
            ev_vowel_time.append([time_2[tock], time_2[tock + 1], time_2[tock + 2]])
        ev_vowel_time.pop(0)

        overlap = [[]]
        pair = [[]]
        for o in fa_vowel_time:
            third_life_fa = (o[1] - o[0]) * (3/4)
            for t in ev_vowel_time:
                if t[0] <= o[0] <= t[1] and t[0] <= t[1] <= o[1] and third_life_fa <= t[1] - o[0]:
                # if t[0] <= o[0] <= t[1] and t[0] <= t[1] <= o[1]:
                    pair.append([o[3], t[2]])
                    overlap.append([o[2], o[3]])
                elif o[0] <= t[0] <= t[1] and t[0] <= o[1] <= t[1] and third_life_fa <= o[1] - t[0]:
                # elif o[0] <= t[0] <= t[1] and t[0] <= o[1] <= t[1]:
                    pair.append([o[3], t[2]])
                    overlap.append([o[2], o[3]])
                elif o[0] <= t[0] < t[1] <= o[1]:
                    pair.append([o[3], t[2]])
                    overlap.append([o[2], o[3]])
                elif t[0] <= o[0] < o[1] <= t[1]:
                    pair.append([o[3], t[2]])
                    overlap.append([o[2], o[3]])
        overlap.pop(0)
        pair.pop(0)
        check_1 = []
        check_2 = []
        ct = 0
        for c in range(len(pair)):
            if pair[c][0] not in check_1:
                check_1.append(pair[c][0])
                if pair[c][1] not in check_2:
                    check_2.append(pair[c][1])
                    ct += 1
        return ct
    except:
        return -1

# ----------------------------------------------------------------------------------------------------------------------#
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
        sr1 = vpe / t
        sr2 = vpee / t
        sr3 = vfa / t
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

# ----------------------------------------------------------------------------------------------------------------------#
file_name_template_1 = 'F:\Projects\Active Projects\Project Intern_IITB\Envelope\\'
result_name = 'Vowel_Evaluation_Flashpoint_STE_O75_50_7.csv'

copyfile('C:\Users\Mahe\PycharmProjects\Internship_IITB\\Flashpoint_STE_Extension.py',
         'F:\Projects\Active Projects\Project Intern_IITB\Envelope\\' + result_name[:-4] + '.txt')

if not os.path.exists(file_name_template_1 + "Analyze\\" + result_name[:-4]):
    os.makedirs(file_name_template_1 + "Analyze\\" + result_name[:-4])
# ----------------------------------------------------------------------------------------------------------------------#
file_no = 1
results_vowel = open(file_name_template_1 + result_name, 'w')  # The csv file where the results are saved
only_audio = glob.glob(file_name_template_1 + 'Data 1\*.wav')  # Extract file name of all audio samples
only_text = glob.glob(
    file_name_template_1 + '\Data 1\*.TextGrid')  # Extract file name of all force aligned TextGrid
results_vowel.write(
    'File No' + ',' + 'Name' + ',' + 'Peak_Elimination' + ',' + 'Evaluation Script PE' + ',' + 'Vowel Count FA' +
    ',' + 'Duration(In Seconds)' + ',' + 'Precision PE' + ',' + 'Recall PE' + ',' + 'Tag' + ',' + 'Speaking rate 1' + ',' +
    'Speaking rate 2' + ',' + 'Speaking rate 3' + ',' + 'Speed 1' + ',' + 'Speed 2' + ',' + 'Speed 3' + '\n')  # Column headers
# ----------------------------------------------------------------------------------------------------------------------#
for j in only_audio:
    filename1, count1 = peak_elimination(j)  # Run Peak elimination algorithm
    textgrid(str(filename1[:-4] + 'PE.csv'))  # Create TextGrid based on results of peak elimination

    count_pe = evaluation(str(j[:-4] + '.TextGrid'), str(
        filename1[:-4] + 'PE_NEW.TextGrid'))  # Evaluate the TextGrid created and the force aligned TextGrid
    vowel_count, time = fa_count(str(j[
                                     :-4] + '.TextGrid'))  # Count the number of vowels in the audio file according to the force aligned TextGrid
    results_write(file_no, j, count1, count_pe, vowel_count, time)  # Writing the results in the csv file.
    # ----------------------------------------------------------------------------------------------------------------------#
    copyfile(j, file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + '.wav')
    copyfile(str(j[:-4] + '.TextGrid'),
             file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'FA.TextGrid')
    copyfile(str(filename1[:-4] + 'PE_NEW.TExtGrid'),
             file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'PE.TextGrid')
    file_no += 1
    sys.stdout.write("\r{0}".format((round(float(file_no / len(only_audio)) * 100, 2))))
    sys.stdout.flush()

print '\n', datetime.now() - startTime  # Print program run time
winsound.Beep(300, 2000)
win32api.MessageBox(0, 'The code has finished running', 'Complete', 0x00001000)

results_vowel.close()
