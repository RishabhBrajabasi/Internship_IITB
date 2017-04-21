from __future__ import division

import csv
import glob
import math
import os
import re
import win32api
import winsound
from datetime import datetime
from math import factorial

import numpy as np
from scipy.io import wavfile

startTime = datetime.now()  # To calculate the run time of the code.

#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------------------------------------#
def peak_elimination(audio, window_dur=50, hop_dur=7, window_number = 1, ste_number = 1):
    new = ''
    replace = str(audio).find('Data 1')
    new = new + (audio[0:replace])
    new += 'Data 2'
    new = new + (audio[replace + 6:])
    text_file_1 = open(str(new[:-4]) + 'PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid
    try:
        audio_file = audio
        fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
        data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
        window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
        hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples

        if window_number == 1:
            window_type = np.bartlett(window_size)  # Window type: Hanning (by default)
        elif window_number == 2:
            window_type = np.blackman(window_size)  # Window type: Hanning (by default)
        elif window_number == 3:
            window_type = np.hamming(window_size)  # Window type: Hanning (by default)
        elif window_number == 4:
            window_type = np.hanning(window_size)  # Window type: Hanning (by default)
        else:
            print 'Error in Windowing'
            window_type = np.hanning(window_size)  # Window type: Hanning (by default)

        no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
        zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
        data = np.concatenate((data, zero_array))
        length = len(data)  # Finding length of the actual data
    #----------------------------------------------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------------------------------------#
        st_energy = []
        if ste_number == 1:
            for i in range(no_frames):
                frame = data[i * hop_size:i * hop_size + window_size] * window_type
                st_energy.append(sum(frame ** 2))

        elif ste_number == 2:
            for i in range(no_frames):
                frame = data[i * hop_size:i * hop_size + window_size] * window_type
                st_energy.append(math.sqrt(sum(frame ** 2) / len(frame)))

        elif ste_number == 3:

            for i in range(no_frames):
                frame = np.abs(data[i * hop_size:i * hop_size + window_size] * window_type)
                st_energy.append(sum(frame))


        elif ste_number == 4:
            for i in range(no_frames):
                frame = np.abs(data[i * hop_size:i * hop_size + window_size] * window_type)
                for j in range(1, len(frame) - 1, 1):
                    frame[j] = frame[j] * frame[j] - frame[j + 1] * frame[j - 1]
                st_energy.append(sum(frame))

        else:
            print 'Error in STE'
            for i in range(no_frames):
                frame = data[i * hop_size:i * hop_size + window_size] * window_type
                st_energy.append(sum(frame ** 2))

        norm_max = max(st_energy)
        for i in range(no_frames):
            st_energy[i] = st_energy[i] / norm_max

    #----------------------------------------------------------------------------------------------------------------------#

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

    #----------------------------------------------------------------------------------------------------------------------#
        mark = []
        for p in range(len(peak)):  # Extracting a 50 ms slice of the audio file based on the frame number
            if peak[p] is not 0:
                mark.append(p * hop_size)
                mark.append(p * hop_size + window_size)

        for i in range(0, len(mark), 2):  # Writing the result to a CSV File
            text_file_1.write('%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + "Vowel" + "\n")

        # mark = []
        # for p in range(len(peak)):  # Extracting a 50 ms slice of the audio file based on the frame number
        #     if peak[p] is not 0:
        #         mark.append(p * hop_size)
        #         mark.append(p * hop_size + window_size)
        # # print mark
        #
        # time = []
        # for i in range(0, len(mark), 2):
        #     time.append(mark[i] * 0.0000625)
        #     time.append(mark[i + 1] * 0.0000625)
        # print time
        # # print time
        # # if time[0] == 0.0:
        # #     text_file_1.write('%06.3f' % (time[0]) + "\t" + '%06.3f' % (time[1]) + "\t" + "Vowel" + "\n")
        # # else:
        # #     text_file_1.write('%06.3f' % (0) + "\t" + '%06.3f' % (time[0]) + "\t" + "NV" + "\n")
        # #
        # #
        # # for i in range(2,len(time)):  # Writing the result to a CSV File
        # #     if i == 2:
        # #         text_file_1.write('%06.3f' % (time[i - 1]) + "\t" + '%06.3f' % (time[i]) + "\t" + "NV" + "\n")
        # #     text_file_1.write('%06.3f' % (time[i]) + "\t" + '%06.3f' % (time[i + 1]) + "\t" + "Vowel" + "\n")
        # #     text_file_1.write('%06.3f' % (time[i+1]) + "\t" + '%06.3f' % (time[i + 2]) + "\t" + "NV" + "\n")
        # for i in range(0, len(time), 2):  # Writing the result to a CSV File
        #     if time[0] == 0.0:
        #         text_file_1.write('%06.3f' % (time[i]) + "\t" + '%06.3f' % (time[i + 1]) + "\t" + "Vowel" + "\n")
        #         text_file_1.write('%06.3f' % (time[i + 1]) + "\t" + '%06.3f' % (time[i + 2]) + "\t" + " " + "\n")
        #     else:
        #         text_file_1.write('%06.3f' % (time[i]) + "\t" + '%06.3f' % (time[i + 1]) + "\t" + " " + "\n")
        #         text_file_1.write('%06.3f' % (time[i + 1]) + "\t" + '%06.3f' % (time[i + 2]) + "\t" + "Vowel" + "\n")
        return new, count_of_peaks
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
        precision.append(vpee / float(vpe))
        recall.append(vpee / float(vfa))
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#

wd = [20]
hd = [3]
wn = [1]
sn = [2]
cases = len(wd)*len(hd)*len(wn)*len(sn)

results_opt = open('F:\Projects\Active Projects\Project Intern_IITB\Baseline Investigation\Confirmation 2.csv', 'a')
if os.path.getsize('F:\Projects\Active Projects\Project Intern_IITB\Baseline Investigation\Vowel_opt_Window and Hop Test.csv') == 0:
    results_opt.write('File Name' + ',' + 'Window Duration' + ',' + 'Hop Duration' + ',' + 'Window Used'
                      + ',' + 'STE Used' + ',' + 'Precision' + ',' + 'Recall' + '\n')

file_name_template_1 = 'F:\Projects\Active Projects\Project Intern_IITB\Baseline Investigation\\'
only_audio = glob.glob(file_name_template_1 + 'Data 1\*.wav')  # Extract file name of all audio samples
only_text = glob.glob(file_name_template_1 + '\Data 1\*.TextGrid')  # Extract file name of all force aligned TextGrid

result_name = []
for name in range(cases):
    result_name.append('Baseline_opt' + str(name+1) + '.csv')

file_no = 1
file_count = 0

precision = []
recall = []


for wdd in wd:
    for hdd in hd:
        for wnn in wn:
            for snn in sn:
                results_vowel = open(file_name_template_1 + result_name[file_count], 'w')  # The csv file where the results are saved
                results_vowel.write('File No' + ',' + 'Name' + ',' + 'Peak_Elimination' + ',' + 'Evaluation Script PE' + ',' + 'Vowel Count FA' + ',' + 'Duration(In Seconds)' + ',' +
                                    'Precision PE' + ',' + 'Recall PE' + ',' + 'Tag' + '\n')  # Column headers
                for j in only_audio:
                    filename1, count1 = peak_elimination(j, window_dur=wdd, hop_dur=hdd, window_number=wnn, ste_number= snn)
                    textgrid(str(filename1[:-4] + 'PE.csv'))
                    count_pe = evaluation(str(j[:-4] + '.TextGrid'), str(filename1[:-4] + 'PE_NEW.TextGrid'))
                    vowel_count, time = fa_count(str(j[:-4] + '.TextGrid'))
                    results_write(file_no, j, count1, count_pe, vowel_count, time)
                    file_no += 1
                results_opt.write(str(result_name[file_count]) + ',' + str(wdd) + ',' + str(hdd) + ',' + str(wnn) + ',' + str(snn) +
                                  ',' + str(sum(precision)/len(precision)) + ',' + str(sum(recall)/len(recall)) + '\n')
                precision[:] = []
                recall[:] = []
                file_count += 1
                file_no = 1

print datetime.now() - startTime
winsound.Beep(300, 2000)
win32api.MessageBox(0, 'The code has finished running', 'Complete', 0x00001000)
