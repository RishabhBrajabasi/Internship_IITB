import csv
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, hilbert

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
audio = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\' + file_no + '.wav'
text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel_Identification\Saitama_PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid
text_file_2 = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel_Identification\Genos.csv', 'w')
text_file_3 = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel_Identification\OnePunch.csv', 'w')

fs, audio_data = wavfile.read(audio)  # Extract the sampling frequency and the data points of the audio file.
audio_data = audio_data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
audio_data = butter_band_pass_filter(audio_data, 300, 2500, fs, order=6)  # Filtering the data.

data_1 = []
for sample_1 in range(len(audio_data)):
    if audio_data[sample_1] < 0:
        data_1.append(audio_data[sample_1] * -1)  # Calculating absolute of the data.
    else:
        data_1.append(audio_data[sample_1])

analytic_signal = hilbert(data_1)
amplitude_envelope = np.abs(analytic_signal)

amplitude_envelope = moving_average(amplitude_envelope, 800)  # Calculating envelope of the audio file
amplitude_envelope = butter_low_pass_filter(amplitude_envelope, 100, fs, 5)  # Filtering out the high frequency ripples in the curve.
amplitude_envelope = moving_average(amplitude_envelope, 20)  # Smoothing the curve.

# data_1 = moving_average(data_1, 1000)  # Calculating envelope of the audio file
# data_1 = butter_low_pass_filter(data_1, 100, fs, 5)  # Filtering out the high frequency ripples in the curve.
# data_1 = moving_average(data_1, 20)  # Smoothing the curve.
#
# max1 = max(data_1)  # Finding the maximum of the curve.
# for sample_2 in range(len(data_1)):
#     data_1[sample_2] = data_1[sample_2] / max1  # Normalizing the curve
# data_1 = data_1.tolist()

max1 = max(amplitude_envelope)  # Finding the maximum of the curve.
for sample_2 in range(len(amplitude_envelope)):
    amplitude_envelope[sample_2] = amplitude_envelope[sample_2] / max1  # Normalizing the curve
data_1 = amplitude_envelope.tolist()
#----------------------------------------------------------------------------------------------------------------------#
"""
Find all maxima's in the envelope of the audio file. Maxima's which have a magnitude above the threshold are
marked as peaks. The magnitude of the peak and the location of the peak is stored.
"""
threshold_peak = 0.1  # Minimum value for a maxima to qualify as a peak.
peak_1 = []
location_peak_1 = []
for p in range(len(data_1)):
    if p == 0:
        peak_1.append(0)
    elif p == len(data_1) - 1:
        peak_1.append(0)
    else:
        if data_1[p] > data_1[p + 1] and data_1[p] > data_1[p - 1] and data_1[p] >= threshold_peak:
            peak_1.append(data_1[p])  # Finding value of the peak.
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
    value = data_1[location_peak_1[l1]] * 0.85
    for ele in range(location_peak_1[l1], -1, -1):
        if data_1[ele] < value:
            flash_1 = ele  # Finding the first flash point, to the left of the peak.
            break
    for ele in range(location_peak_1[l1], len(data_1), 1):
        if data_1[ele] < value:
            flash_2 = ele  # Finding the second flash point, to the right of the peak.
            break
    the_list_1.append([flash_1, location_peak_1[l1], flash_2, count, data_1[location_peak_1[l1]]])
    count += 1
the_list_1.pop(0)

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
#----------------------------------------------------------------------------------------------------------------------#
"""
Between a pair of two peaks, the point with the least magnitude is found and is marked as a valley.
The staring point and the last point of the envelope are also marked as valleys. If the valley is located within
the region of the flash points of the peak, the peak is removed. Once the shortlisted peaks are obtained, valleys
between peaks are found again as mentioned earlier.
"""
valley_1 = [0]
for v1 in range(len(the_list_2) - 1):
    minimum = min(data_1[the_list_2[v1][1]:the_list_2[v1 + 1][1]])  # Region between the two peaks
    valley_1.append(data_1.index(minimum))
valley_1.append(len(data_1) - 1)

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
    minimum = min(data_1[peak_2[v]:peak_2[v + 1]])
    valley_2.append(data_1.index(minimum))
valley_2.append(len(data_1) - 1)
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
    value = data_1[peak_2[l3]] * 0.707
    for ele in range(peak_2[l3], -1, -1):
        if data_1[ele] < value:
            flash_1 = ele
            break
    for ele in range(peak_2[l3], len(data_1), 1):
        if data_1[ele] < value:
            flash_2 = ele
            break
    the_list_4.append([valley_2[l3], flash_1, peak_2[l3], flash_2, valley_2[l3 + 1]])
the_list_4.pop(0)

boundary = [[]]
vowel_boundary = [[]]
for b in the_list_4:
    if b[0] < b[1] < b[3] < b[4]:  # valley_left < flash_left < flash_right < valley_right
        boundary.append([b[1], b[3]])  # flash_left <--> flash_right
        vowel_boundary.append([b[1], b[3]])

    elif b[0] < b[1] and b[3] > b[4]:  # valley_left < flash_left and flash_right > valley_right
        boundary.append([b[1], b[4]])  # flash_left <--> valley_right
        vowel_boundary.append([b[1], b[4]])

    elif b[0] > b[1] and b[3] < b[4]:  # valley_left > flash_left and flash_right < valley_right
        boundary.append([b[0], b[3]])  # valley_left <--> flash_right
        vowel_boundary.append([b[0], b[3]])

    elif b[0] > b[1] and b[3] > b[4]:  # valley_left < flash_left and flash_right > valley_right
        boundary.append([b[0], b[4]])  # valley_left <--> valley_right
        vowel_boundary.append([b[0], b[4]])
    else:
        boundary.append([b[0], b[4]])  # valley_left <--> valley_right
        vowel_boundary.append([b[0], b[4]])
boundary.pop(0)
vowel_boundary.pop(0)


for b_value in range(len(boundary)):
    boundary[b_value][0] = round(boundary[b_value][0] / float(fs), 2)
    boundary[b_value][1] = round(boundary[b_value][1] / float(fs), 2)
    #----------------------------------------------------------------------------------------------------------------------#
"""
Enhancing the boundaries to mark vowel and non- vowel centres. Storing the result in a  text_file for .TextGrid creation.
"""
x_values = np.arange(0, len(audio_data), 1) / float(fs)

better_boundaries = [[0.00, boundary[0][0]]]
for bb in range(len(boundary) - 1):
    better_boundaries.append([boundary[bb][0], boundary[bb][1]])
    better_boundaries.append([boundary[bb][1], boundary[bb + 1][0]])
better_boundaries.append([boundary[-1][0], boundary[-1][1]])
better_boundaries.append([boundary[-1][1], round(x_values[-1], 2)])

for tf in range(len(better_boundaries)):
    if tf % 2 == 0:
        text_file_1.write('%06.3f' % (better_boundaries[tf][0]) + "\t" + '%06.3f' % (better_boundaries[tf][1]) + "\t" + " " + "\n")
    else:
        text_file_1.write('%06.3f' % (better_boundaries[tf][0]) + "\t" + '%06.3f' % (better_boundaries[tf][1]) + "\t" + "Vowel" + "\n")
text_file_1.close()

csvFileName = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel_Identification\Saitama_PE.csv'
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
def filter_bank(o_data, low_pass, high_pass, fs, order_of_filter, vowel_length):
    atad = butter_band_pass_filter(o_data, low_pass, high_pass, fs, order_of_filter)
    # window_type = np.hanning(vowel_length)
    energy = []
    for i in range(vowel_length):
        # atad[i] = atad[i] * window_type[i]
        energy.append(atad[i]*atad[i])
    return sum(energy)

vs = []
vowel_count = 0
taakat = [[]]
the_Matrix = [[]]

for vb in vowel_boundary:
    length_vowel = vb[1] - vb[0]
    list_energy = [[]]

    st_energy_1 = filter_bank(audio_data[vb[0]:vb[1]], 100, 500, fs, 6, length_vowel)
    list_energy.append([vowel_count, 'B1', st_energy_1])
    st_energy_2 = filter_bank(audio_data[vb[0]:vb[1]], 500, 1000, fs, 6, length_vowel)
    list_energy.append([vowel_count, 'B2', st_energy_2])
    st_energy_3 = filter_bank(audio_data[vb[0]:vb[1]], 1000, 2500, fs, 6, length_vowel)
    list_energy.append([vowel_count, 'B3', st_energy_3])
    st_energy_4 = filter_bank(audio_data[vb[0]:vb[1]], 2500, 4000, fs, 6, length_vowel)
    list_energy.append([vowel_count, 'B4', st_energy_4])
    st_energy_5 = filter_bank(audio_data[vb[0]:vb[1]], 4000, 5000, fs, 6, length_vowel)
    list_energy.append([vowel_count, 'B5', st_energy_5])
    st_energy_6 = filter_bank(audio_data[vb[0]:vb[1]], 5000, 6000, fs, 6, length_vowel)
    list_energy.append([vowel_count, 'B6', st_energy_6])
    st_energy_7 = filter_bank(audio_data[vb[0]:vb[1]], 6000, 7500, fs, 6, length_vowel)
    list_energy.append([vowel_count, 'B7', st_energy_7])
    list_energy.pop(0)
    zero_array = np.zeros(1000)  # Appending appropriate number of zeros
    goku = audio_data[vb[0]:vb[1]]
    goku = goku.tolist()
    the_Matrix.append(list_energy)
    vs = vs + goku
    zero_array = zero_array.tolist()
    vs = vs + zero_array
    taakat.append([vowel_count, ['B1', st_energy_1], ['B2', st_energy_2], ['B3', st_energy_3], ['B4', st_energy_4], ['B5', st_energy_5],
                   ['B6', st_energy_6], ['B7', st_energy_7]])
    vowel_count += 1
taakat.pop(0)
the_Matrix.pop(0)

name = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel_Identification\\vowel_segment_apple' + str(9000) + '.wav'
vs = np.array(vs)
wavfile.write(name, 16000, vs)

for t in taakat:
    text_file_2.write(str(t[0]) + "," + str(t[1]) + "," + str(t[2]) + "," + str(t[3]) + "," + str(t[4]) + "," + str(t[5]) + "," + str(t[6]) + "," + str(t[7]) + "\n")

descending_order_of_taakat = [[]]
for row in range(len(the_Matrix)):
    descending_order_of_taakat.append(sorted(the_Matrix[row], key=lambda student: student[2], reverse=1))
descending_order_of_taakat.pop(0)
band_energy = [[]]

for band in range(len(descending_order_of_taakat)):
    band_energy.append([descending_order_of_taakat[band][0][1], descending_order_of_taakat[band][1][1], descending_order_of_taakat[band][2][1],
                        descending_order_of_taakat[band][3][1], descending_order_of_taakat[band][4][1], descending_order_of_taakat[band][5][1],
                        descending_order_of_taakat[band][6][1]])
band_energy.pop(0)
print band_energy

v_count = 0
for be in band_energy:
    text_file_3.write(str(v_count) + ',' + str(be[0]) + ',' + str(be[1]) + ',' + str(be[2]) + ',' + str(be[3]) + ',' +
                      str(be[4]) + ',' + str(be[5]) + ',' + str(be[6]) + '\n')
    v_count += 1

# import praatUtil
# import os
# from matplotlib import pyplot as plt
# import matplotlibUtil
# import generalUtility
# import sys