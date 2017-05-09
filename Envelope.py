from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import re

file_no = '17'
audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + '.wav'
textgridFA = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + 'FA.TextGrid'
text_grid_1 = open(textgridFA, 'r')  # Open the FA TextGrid
data_1 = text_grid_1.read()  # Read and assign the content of the FA TextGrid to data_1
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
        time_1.append((data_1[m.start() + 8]))
    elif data_1[m.start() + 10] == '"':
        time_1.append((data_1[m.start() + 8] + data_1[m.start() + 9]))
    else:
        time_1.append((data_1[m.start() + 8] + data_1[m.start() + 9] + data_1[m.start() + 10]))
    time_1.append(counter)
    counter += 1
#----------------------------------------------------------------------------------------------------------------------#
def butter_bandpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a

def butter_bandpass_filter(data, highcut, fs, order=5):
    b, a = butter_bandpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    plt.show()
    return np.convolve(interval, window, 'same')

fs, data = wavfile.read(audio_file)
data = data / float(2 ** 15)
x_values = np.arange(0, len(data), 1) / float(fs)
batman = data
max_batman = max(batman)
for k in range(len(batman)):
    batman[k] = batman[k]/max_batman

data1 = []
for i in range(len(data)):
    if data[i] < 0:
        data1.append(data[i]*-1)
    else:
        data1.append(data[i])

max1 = max(data1)
for k in range(len(data1)):
    data1[k] = data1[k]/max1

d1 = moving_average(data1, 1000)
d1 = butter_bandpass_filter(d1, 100, fs, 5)
d1 = moving_average(d1, 20)

max1 = max(d1)
for k in range(len(d1)):
    d1[k] = d1[k]/max1
d1 = d1.tolist()

st_energy_peak = d1
threshold_peak = 0.1

peak_f = []
location_peak_f = []
for p in range(len(st_energy_peak)):
    if p == 0:
        peak_f.append(0)
    elif p == len(st_energy_peak) - 1:
        peak_f.append(0)
    else:
        if st_energy_peak[p] > st_energy_peak[p + 1] and st_energy_peak[p] > st_energy_peak[p - 1] and st_energy_peak[p] >= threshold_peak:
            peak_f.append(st_energy_peak[p])
            location_peak_f.append(p)
        else:
            peak_f.append(0)

the_list = [[]]
the_list_value = [[]]
count = 0
for lines in range(len(location_peak_f)):
    value = d1[location_peak_f[lines]]*0.85
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
    for k in range(l+1, len(s)):
        if s[l][0] < s[k][1] < s[l][2] and s[k][3] not in remove:
            remove.append(s[k][3])

superman = [[]]
for ww in the_list_value:
    if ww[3] not in remove:
        superman.append(ww)
superman.pop(0)

cyborg = [[]]
valley = [0]
for arrow in range(len(superman)-1):
    minimum = min(d1[superman[arrow][1]:superman[arrow+1][1]])
    location_minimum = d1.index(minimum)
    valley.append(location_minimum)
valley.append(len(d1)-1)

fresh_count = 0
for cc in range(len(superman)):
    cyborg.append([valley[cc], superman[cc][0], superman[cc][1], superman[cc][2], valley[cc+1], fresh_count])
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
for v in range(len(star_fire)-1):
    minimum = min(d1[star_fire[v]:star_fire[v+1]])
    val.append(d1.index(minimum))
val.append(len(d1)-1)


plt.subplot(211)
for p in range(len(star_fire)):
    plt.vlines(star_fire[p], 0, d1[star_fire[p]], 'red', linewidth='2.0')

    # value = d1[star_fire[p]] * 0.707
    # for ele in range(star_fire[p], -1, -1):
    #     if d1[ele] < value:
    #         flash_1 = ele
    #         break
    # for ele in range(star_fire[p], len(d1), 1):
    #     if d1[ele] < value:
    #         flash_2 = ele
    #         break
    # plt.hlines(d1[star_fire[p]]*0.707, flash_1, flash_2)
# for v in range(len(val)):
#     plt.vlines(val[v], 0, d1[val[v]], 'green', linewidth='2.0')

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
    martian_man_hunter.append([val[mmh],flash_1,star_fire[mmh],flash_2,val[mmh+1]])

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

for j in range(len(boundary)):
    plt.vlines(boundary[j][0], min(d1), (d1[boundary[j][0]]), color='black', linewidth='2.0')  # Syllable Boundaries
    plt.vlines(boundary[j][1], min(d1), (d1[boundary[j][1]]), color='black', linewidth='2.0')  # Syllable Boundaries


for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j]*fs, min(d1), max(d1), linestyles=':', color='black', linewidth='2.0')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2]*fs, min(d1), time_1[j], fontsize=10, color='black', rotation=0)  # Syllable Labels


plt.plot(d1, 'black', label='Envelope', linewidth='2.0')

for i in range(len(boundary)):
    boundary[i][0] = round(boundary[i][0]/float(fs), 2)
    boundary[i][1] = round(boundary[i][1]/float(fs), 2)

plt.subplot(212)
plt.plot(data)
for j in range(0, len(time_1), 4):
    plt.vlines(time_1[j]*fs, min(data), max(data), linestyles=':', color='black', linewidth='2.0')  # Syllable Boundaries
for j in range(2, len(time_1), 4):
    plt.text(time_1[j - 2]*fs, min(data), time_1[j], fontsize=10, color='black', rotation=0)  # Syllable Labels

x_values = np.arange(0, len(data), 1) / float(fs)
better_boundaries = [[0.00, boundary[0][0]]]
for i in range(len(boundary)-1):
    better_boundaries.append([boundary[i][0], boundary[i][1]])
    better_boundaries.append([boundary[i][1], boundary[i+1][0]])
better_boundaries.append([boundary[-1][0], boundary[-1][1]])
better_boundaries.append([boundary[-1][1], round(x_values[-1], 2)])

text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Envelope\Saitama_PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid
for i in range(len(better_boundaries)):
    if i % 2 == 0:
        text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + " " + "\n")
    else:
        text_file_1.write('%06.3f' % (better_boundaries[i][0]) + "\t" + '%06.3f' % (better_boundaries[i][1]) + "\t" + "Vowel" + "\n")
text_file_1.close()

import csv

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
plt.show()
