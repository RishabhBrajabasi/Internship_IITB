from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

audio_file ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Analyze\Vowel_Evaluation_V4_I2\\17.wav'

def peaks(st_energy_peak, threshold_peak):
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
    return peak_f, location_peak_f
#----------------------------------------------------------------------------------------------------------------------#
def valleys(st_energy_valley, threshold_valley):
    valley_f = []
    location_valley_f = []
    for v in range(len(st_energy_valley)):
        if v == 0:
            if st_energy_valley[v] < st_energy_valley[v + 1] and st_energy_valley[v] < threshold_valley:
                valley_f.append(st_energy_valley[v])
                location_valley_f.append(v)
            else:
                valley_f.append(0)
        elif v == len(st_energy_valley) - 1:
            if st_energy_valley[v] < st_energy_valley[v - 1] and st_energy_valley[v] < threshold_valley:
                valley_f.append(st_energy_valley[v])
                location_valley_f.append(v)
            else:
                valley_f.append(0)
        else:
            if st_energy_valley[v] < st_energy_valley[v + 1] and st_energy_valley[v] < st_energy_valley[v - 1] and st_energy_valley[v] < threshold_valley:
                valley_f.append(st_energy_valley[v])
                location_valley_f.append(v)
            else:
                valley_f.append(0)
    return valley_f, location_valley_f
#----------------------------------------------------------------------------------------------------------------------#
def peak_valley_elimination(location_peak_pve, location_valley_pve):
    location_combined = location_peak_pve + location_valley_pve
    location_combined.sort()
    peak_valley_pair = [[]]
    peak_valley_pair_value = [[]]
    print location_peak_pve
    print location_valley_pve
    print location_combined
    for ele in range(len(location_combined)):
        if location_combined[ele] in location_peak_pve:
            peak_valley_pair.append([location_combined[ele - 1], location_combined[ele], location_combined[ele + 1]])
            peak_valley_pair_value.append([d1[location_combined[ele - 1]], d1[location_combined[ele]], d1[location_combined[ele + 1]]])
    peak_valley_pair.pop(0)
    peak_valley_pair_value.pop(0)
    return peak_valley_pair, peak_valley_pair_value
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
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    plt.show()
    return np.convolve(interval, window, 'same')
#----------------------------------------------------------------------------------------------------------------------#
fs, data = wavfile.read(audio_file)
data = data / float(2 ** 15)
x_values = np.arange(0, len(data), 1) / float(fs)
batman = data
max_batman = max(batman)
for k in range(len(batman)):
    batman[k] = batman[k]/max_batman
#----------------------------------------------------------------------------------------------------------------------#
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
peak, location_peak = peaks(d1, 0.2)
valley, location_valley = valleys(d1, 0.8)
peak_valley, peak_valley_value = peak_valley_elimination(location_peak, location_valley)
print peak_valley
print peak_valley_value

#
# d2 = []
# for i in range(len(d1)):
#     d2.append(round(d1[i], 2))

# print 'D1:', d1[0:100]
# print 'D2:', d2[0:100]
# eliminate = []
# for val in range(len(location_peak)):
#     if d1[peak_valley[val][1]] - d1[peak_valley[val][0]] < 0.05 and d1[peak_valley[val][1]] - d1[peak_valley[val][2]] < 0.05:
#         # print peak_valley[val]
#         # print d1[peak_valley[val][0]], d1[peak_valley[val][1]], d1[peak_valley[val][2]]
#         eliminate.append(peak_valley[val])
#     if peak_valley[val][1] - peak_valley[val][0] < 20 or peak_valley[val][2] - peak_valley[val][1] < 20:
#         eliminate.append(peak_valley[val])
# peak_valley_new = [[]]
# for val in range(len(location_peak)):
#     if peak_valley[val] not in eliminate:
#         peak_valley_new.append(peak_valley[val])
# peak_valley_new.pop(0)

# total = 0
# count = 0
# for lines in range(len(peak_valley_new)):
#     plt.vlines(peak_valley_new[lines][0], min(d1), d1[peak_valley_new[lines][0]], 'green')
#     plt.vlines(peak_valley_new[lines][1], min(d1), d1[peak_valley_new[lines][1]], 'black')
#     plt.vlines(peak_valley_new[lines][2], min(d1), d1[peak_valley_new[lines][2]], 'green')
    # total += d1[peak_valley_new[lines][1]]
    # count += 1
# for lines in range(len(peak_valley)):
#     plt.vlines(peak_valley[lines][0], min(d1), d1[peak_valley[lines][0]], 'green')
#     plt.hlines(d1[peak_valley[lines][1]]-0.2, 0, len(d1))
#     plt.vlines(peak_valley[lines][1], min(d1), d1[peak_valley[lines][1]], 'black')
#     plt.vlines(peak_valley[lines][2], min(d1), d1[peak_valley[lines][2]], 'green')
# print type(d1)


value = d1[peak_valley[0][1]]*0.8
flash_1 = min(enumerate(d1[peak_valley[0][0]:peak_valley[0][1]]), key=lambda x: abs(x[1]-value))
print 'Flash_1:', flash_1

flash_2 = min(enumerate(d1[peak_valley[0][1]:peak_valley[0][2]]), key=lambda x: abs(x[1]-value))
print 'Flash_2:', flash_2

print type(peak_valley[0][0])

print 'Peak Value:', d1[peak_valley[0][1]]
print 'Value', value
print peak_valley[0][0], peak_valley[0][1], peak_valley[0][2]
print d1[peak_valley[0][0]], d1[peak_valley[0][1]], d1[peak_valley[0][2]]
print d1[peak_valley[0][0]:peak_valley[0][2]]
# x1 = d1.index(value)
plt.plot(d1, 'black', label='Envelope', linewidth='2.0')
plt.hlines(d1[peak_valley[0][1]] - 0.2, flash_1[0] + peak_valley[0][0], flash_2[0] + peak_valley[0][1], 'red', linewidth='2.0' )
plt.show()
# # plt.hlines(total/count, 0, len(d1))

# # plt.plot(batman, 'blue', label='Audio')
# plt.plot(d1, 'black', label='Envelope', linewidth='2.0')
# plt.plot(d2, 'red', label='Envelope', linewidth='2.0')
# # plt.hlines(0.1, 0, len(d1))
# # plt.hlines(0.8, 0, len(d1))
#
# plt.legend(loc='best')
# plt.show()
