from scipy.io import wavfile
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from math import factorial
from scipy.signal import butter, lfilter

audio_file ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Analyze\Vowel_Evaluation_V4_I2\\17.wav'
window_dur = 30
hop_dur = 5

point1 = (2, 5)
point2 = (7, 8)

plt.plot(point1, point2)
plt.show()


fs, data = wavfile.read(audio_file)
data = data / float(2 ** 15)
window_size = int(window_dur * fs * 0.001)
hop_size = int(hop_dur * fs * 0.001)
window_type = np.hanning(window_size)
no_frames = int(math.ceil(len(data) / (float(hop_size))))
zero_array = np.zeros(window_size)
data = np.concatenate((data, zero_array))
length = len(data)
x_values = np.arange(0, len(data), 1) / float(fs)
# plt.plot(x_values, data)
# plt.show()

p = []
l = []
for j in range(len(data)):
    if data[j] <= 0:
        p.append(data[j]*0)
    else:
        p.append(data[j])
loc = []
t = ()
last = p[0]
for j in range(1, len(p)-1):
    if p[j] > p[j+1] and p[j] > p[j-1] and p[j] > last:
        l.append(p[j])
        loc.append(j)
        last = p[j]
        t = t + (p[j], j)

for i in range(len(l)-1):
    p1 = (loc[i], l[i])
    p2 = (loc[i+1], l[i+1])
    print p1, p2
    plt.plot(p1, p2)
plt.show()

plt.plot(loc, l)
plt.show()
# print (l ,loc)
# print t
# plt.plot(x_values, p)
# plt.show()
# d = []
# for i in range(len(data)):
#     d.append(data[i] * data[i])
#
# plt.plot(x_values, d)
# plt.show()

# plt.scatter(l, p)
# plt.plot(l, p)
# plt.show()
