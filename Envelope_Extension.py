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
import sys
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
def butter_bandpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a
#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass_filter(data, highcut, fs, order=5):
    b, a = butter_bandpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#----------------------------------------------------------------------------------------------------------------------#
def peak_elimination(audio):
    new = ''
    start = audio.find('Data 1')
    new = new + (audio[0:start + 5])
    new += '2\\'
    new = new + (audio[start+8:])
    text_file_1 = open(str(new[:-4]) + 'PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid
    try:
        fs, data = wavfile.read(audio)
        data = data / float(2 ** 15)
        x_values = np.arange(0, len(data), 1) / float(fs)
        batman = data
        max_batman = max(batman)
        for k in range(len(batman)):
            batman[k] = batman[k] / max_batman

        data1 = []
        for i in range(len(data)):
            if data[i] < 0:
                data1.append(data[i] * -1)
            else:
                data1.append(data[i])

        max1 = max(data1)
        for k in range(len(data1)):
            data1[k] = data1[k] / max1

        d1 = moving_average(data1, 1000)
        d1 = butter_bandpass_filter(d1, 100, fs, 5)
        d1 = moving_average(d1, 20)

        max1 = max(d1)
        for k in range(len(d1)):
            d1[k] = d1[k] / max1
        d1 = d1.tolist()

        st_energy_peak = d1
        threshold_peak = 0.1
        # ----------------------------------------------------------------------------------------------------------------------#
        peak_f = []
        location_peak_f = []
        for p in range(len(st_energy_peak)):
            if p == 0:
                peak_f.append(0)
            elif p == len(st_energy_peak) - 1:
                peak_f.append(0)
            else:
                if st_energy_peak[p] > st_energy_peak[p + 1] and st_energy_peak[p] > st_energy_peak[p - 1] and \
                                st_energy_peak[p] >= threshold_peak:
                    peak_f.append(st_energy_peak[p])
                    location_peak_f.append(p)
                else:
                    peak_f.append(0)
        # ----------------------------------------------------------------------------------------------------------------------#
        the_list = [[]]
        the_list_value = [[]]
        count = 0
        for lines in range(len(location_peak_f)):
            value = d1[location_peak_f[lines]] * 0.85
            for ele in range(location_peak_f[lines], -1, -1):
                if d1[ele] < value:
                    flash_1 = ele
                    break
            for ele in range(location_peak_f[lines], len(d1), 1):
                if d1[ele] < value:
                    flash_2 = ele
                    break
            the_list.append([flash_1, location_peak_f[lines], flash_2, count])
            the_list_value.append([flash_1, location_peak_f[lines], flash_2, count, d1[location_peak_f[lines]]])
            count += 1

        the_list.pop(0)
        the_list_value.pop(0)
        s = sorted(the_list_value, key=lambda student: student[4], reverse=1)

        remove = []
        for l in range(len(s)):
            for k in range(l + 1, len(s)):
                if s[l][0] < s[k][1] < s[l][2] and s[k][3] not in remove:
                    remove.append(s[k][3])

        superman = [[]]
        for ww in the_list_value:
            if ww[3] not in remove:
                superman.append(ww)
        superman.pop(0)

        cyborg = [[]]
        valley = [0]
        for arrow in range(len(superman) - 1):
            minimum = min(d1[superman[arrow][1]:superman[arrow + 1][1]])
            location_minimum = d1.index(minimum)
            valley.append(location_minimum)
        valley.append(len(d1) - 1)

        fresh_count = 0
        for cc in range(len(superman)):
            cyborg.append([valley[cc], superman[cc][0], superman[cc][1], superman[cc][2], valley[cc + 1], fresh_count])
            fresh_count += 1
        cyborg.pop(0)

        remove[:] = []
        for cc in cyborg:
            if cc[3] > cc[4] or cc[0] > cc[1]:
                remove.append(cc[5])

        star_fire = []
        for sf in cyborg:
            if sf[5] not in remove:
                star_fire.append(sf[2])

        val = [0]
        for v in range(len(star_fire) - 1):
            minimum = min(d1[star_fire[v]:star_fire[v + 1]])
            val.append(d1.index(minimum))
        val.append(len(d1) - 1)

        martian_man_hunter = [[]]
        for mmh in range(len(star_fire)):
            value = d1[star_fire[mmh]] * 0.707
            for ele in range(star_fire[mmh], -1, -1):
                if d1[ele] < value:
                    flash_1 = ele
                    break
            for ele in range(star_fire[mmh], len(d1), 1):
                if d1[ele] < value:
                    flash_2 = ele
                    break
            martian_man_hunter.append([val[mmh], flash_1, star_fire[mmh], flash_2, val[mmh + 1]])
        martian_man_hunter.pop(0)

        boundary = [[]]
        for b in martian_man_hunter:
            if b[0] < b[1] < b[3] < b[4]:
                boundary.append([b[1], b[3]])
            elif b[0] < b[1] and b[3] > b[4]:
                boundary.append([b[1], b[4]])
            elif b[0] > b[1] and b[3] < b[4]:
                boundary.append([b[0], b[3]])
            elif b[1] > b[0] and b[3] > b[4]:
                boundary.append([b[0], b[4]])
            else:
                boundary.append([b[0], b[4]])
        boundary.pop(0)

        for i in range(len(boundary)):
            boundary[i][0] = round(boundary[i][0] / float(fs), 2)
            boundary[i][1] = round(boundary[i][1] / float(fs), 2)

        # ----------------------------------------------------------------------------------------------------------------------#
        x_values = np.arange(0, len(data), 1) / float(fs)
        better_boundaries = [[0.00, boundary[0][0]]]
        for i in range(len(boundary) - 1):
            better_boundaries.append([boundary[i][0], boundary[i][1]])
            better_boundaries.append([boundary[i][1], boundary[i + 1][0]])
        better_boundaries.append([boundary[-1][0], boundary[-1][1]])
        better_boundaries.append([boundary[-1][1], round(x_values[-1], 2)])

        for i in range(len(better_boundaries)):
            if i % 2 == 0:
                text_file_1.write(
                    '%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + " " + "\n")
            else:
                text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (
                better_boundaries[i][1]) + "\t" + "Vowel" + "\n")
        text_file_1.close()

        return new, len(boundary)
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
        vowel_data = ['aa', 'AA', 'ae', 'aw', 'ay', 'ee', 'ex', 'ii', 'II', 'oo', 'OO', 'oy', 'uu', 'UU']
        fa_vowel_time = [[]]
        for tick in range(0, len(time_1), 4):
            if time_1[tick + 2] in vowel_data:
                fa_vowel_time.append([time_1[tick], time_1[tick+1], time_1[tick+2], time_1[tick+3]])
        fa_vowel_time.pop(0)
        # print file_no
        # print '\nFA vowel T.I.M.E', fa_vowel_time

        ev_vowel_time = [[]]
        for tock in range(0, len(time_2), 2):
            ev_vowel_time.append([time_2[tock], time_2[tock+1]])
        ev_vowel_time.pop(0)
        # print 'EV vowel T.I.M.E', ev_vowel_time

        overlap = [[]]
        for o in fa_vowel_time:
            for t in ev_vowel_time:
                if t[0] <= o[0] <= t[1] and t[0] <= t[1] <= o[1]:
                    # print t[0], '<=', o[0], '<=', t[1], 'and', t[0], '<=', t[1], '<=', o[1]
                    overlap.append([o[2], o[3]])
                elif o[0] <= t[0] <= t[1] and t[0] <= o[1] <= t[1]:
                    # print o[0], '<=', t[0], '<=', t[1], 'and', t[0], '<=', o[1], '<=', t[1]
                    overlap.append([o[2], o[3]])
                elif o[0] <= t[0] < t[1] <= o[1]:
                    # print o[0], '<=', t[0], '<', t[1], '<=', o[1]
                    overlap.append([o[2], o[3]])
                elif t[0] <= o[0] < o[1] <= t[1]:
                    # print t[0], '<=', o[0], '<=', o[1], '<=', t[1]
                    overlap.append([o[2], o[3]])
        overlap.pop(0)
        # print 'Overlap', overlap
        # print time_1
        # print time_2
        unique_count = 0
        check = []
        for lap in overlap:
            if lap[1] not in check:
                check.append(lap[1])
        # print len(overlap), len(check)
        # for outer in range(0, len(time_2), 2):
        #     for inner in range(0, len(time_1), 4):
        #         # print  time_1[inner], '<=', time_2[outer], '<', time_1[inner + 1], 'and', time_1[inner], '<', time_2[outer + 1], '<=', time_1[
        #         #             inner + 1]
        #         if time_1[inner] <= time_2[outer] < time_1[inner + 1] and time_1[inner] < time_2[outer + 1] <= time_1[
        #                     inner + 1]:
        #             listing.append(time_1[inner + 2])
        #             listing.append(time_1[inner + 3])
        #
        # for outer in range(0, len(time_2), 2):
        #     for inner in range(0, len(time_1) - 4, 4):
        #         if time_1[inner] <= time_2[outer] <= time_1[inner + 1] and time_1[inner + 4] <= time_2[outer + 1] <= time_1[
        #                     inner + 5]:
        #             listing.append(time_1[inner + 2])
        #             listing.append(time_1[inner + 3])
        #             listing.append(time_1[inner + 6])
        #             listing.append(time_1[inner + 7])
        # print 'Listing', listing
        # count = 0
        # vowel_data = ['aa', 'AA', 'ae', 'aw', 'ay', 'ee', 'ex', 'ii', 'II', 'oo', 'OO', 'oy', 'uu', 'UU']
        #
        # already_here = []
        # for vowel_sound in range(0, len(listing), 2):
        #     if listing[vowel_sound] in vowel_data and listing[vowel_sound + 1] not in already_here:
        #         count += 1
        #         already_here.append(listing[vowel_sound + 1])
        # print 'Already Here', already_here
        return len(check)
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
file_name_template_1 = 'F:\Projects\Active Projects\Project Intern_IITB\Envelope\\'
result_name = 'Vowel_Evaluation_4.csv'

copyfile('C:\Users\Mahe\PycharmProjects\Internship_IITB\\Envelope_Extension.py',
         'F:\Projects\Active Projects\Project Intern_IITB\Envelope\\' + result_name[:-4] + '.txt')

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
    filename1, count1 = peak_elimination(j)  # Run Peak elimination algorithm
    textgrid(str(filename1[:-4] + 'PE.csv'))  # Create TextGrid based on results of peak elimination
    # print file_no
    count_pe = evaluation(str(j[:-4] + '.TextGrid'), str(filename1[:-4] + 'PE_NEW.TextGrid'))  # Evaluate the TextGrid created and the force aligned TextGrid
    vowel_count, time = fa_count(str(j[:-4] + '.TextGrid'))  # Count the number of vowels in the audio file according to the force aligned TextGrid
    results_write(file_no, j, count1, count_pe, vowel_count, time)  # Writing the results in the csv file.
    # ----------------------------------------------------------------------------------------------------------------------#
    copyfile(j, file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + '.wav')
    copyfile(str(j[:-4] + '.TextGrid'), file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'FA.TextGrid')
    copyfile(str(filename1[:-4] + 'PE_NEW.TExtGrid'), file_name_template_1 + "Analyze\\" + result_name[:-4] + "\\" + str(file_no) + 'PE.TextGrid')
    file_no += 1
    sys.stdout.write("\r{0}".format((round(float(file_no / len(only_audio)) * 100, 2))))
    sys.stdout.flush()

print '\n', datetime.now() - startTime  # Print program run time
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
