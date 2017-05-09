from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Analyze\Vowel_Evaluation_V4_I2\\17.wav'

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
threshold_peak = 0.2

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

for p in range(len(star_fire)):
    plt.vlines(star_fire[p], 0, d1[star_fire[p]], 'red', linewidth='2.0')
for v in range(len(val)):
    plt.vlines(val[v], 0, d1[val[v]], 'green', linewidth='2.0')

plt.plot(d1, 'black', label='Envelope', linewidth='2.0')
plt.show()
