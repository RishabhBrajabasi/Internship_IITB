""" This script can be used to find thw vowel candidates in an audio file

Input argument: 1. Complete path of the wav file

Output : A csv file storing the starting and ending time stamps of the vowel sound. Use the TextGrid script to
generate a text grid file which can be viewed in Praat.

Author: Rishabh Brajabasi
Last updated: 25th January 2017
"""

from scipy.io import wavfile
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np





def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

########################################################################################################################
audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Test_accuracy.wav'
window_dur = 50  # Duration of window in milliseconds
hop_dur = 5 # Hop duration in milliseconds
fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
data = data / float(2 ** 15) # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data
########################################################################################################################

noise_energy = 0
energy = [0] * length

#Squaring each data point
for j in range(length):
    energy[j] = data[j] * data[j]

#Calcilating noise energy
for j in range(0, 800):  # energy
    noise_energy += energy[j]
noise_energy /= 800

########################################################################################################################
# points = tuple()
#
# for j in range(len(data)):
#     points = np.append([data[j]], [j])
# print points
#
# hull = ConvexHull(points)
# #
# # plt.plot(points[:,0], points[:,1], 'o')
# # for simplex in hull.simplices:
# #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# #
# # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
# # plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
# plt.plot(hull)
# plt.show()

########################################################################################################################
st_energy = [0] * no_frames
maximum = [0] * no_frames
frame_number = [0] * no_frames
start = [0] * no_frames
stop = [0] * no_frames

#When you want to plot all the frames, Run it sparingly
# fileNameTemplate = r'F:\Projects\Active Projects\Project Intern_IITB\PlotFrame\Plot{0:02d}.png'

#Calculating frame wise short term energy
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    start[i] = i * hop_size
    stop[i] = i * hop_size + window_size
    st_energy[i] = sum(frame ** 2)

    # square = []
    # for j in range(len(frame)):
    #     square.append(frame[j] * frame[j])
    # yhat = savitzky_golay(square, 51, 3)  # window size 51, polynomial order 3
    # Code to individually plot each of the frames created. Since, 4097 plots will be created, run it sparingly
    # plt.plot(yhat)
    # plt.savefig(fileNameTemplate.format(i), format='png')
    # plt.clf()
    # square[:] = []
########################################################################################################################
text_file_4 = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\R3_Mark.csv', 'w')


original_st_energy = st_energy
st_energy = savitzky_golay(st_energy, 51, 3) # window size 51, polynomial order 3

peak = []
count_of_peaks = 0

peak.append(0)
for j in range(1, no_frames - 1):
    if st_energy[j] > st_energy[j + 1] and st_energy[j] > st_energy[j - 1]:
        peak.append(st_energy[j])
        count_of_peaks += 1
    else:
        peak.append(0)
peak.append(0)

print "Number of peaks                        :", count_of_peaks

threshold = 0.04 * (noise_energy + (sum(peak) / count_of_peaks))

count_of_peaks_threshold = 0
peak_threshold = []
location_peak = []
value_peak = []
for j in range(len(peak)):
    if threshold < peak[j]:
        peak_threshold.append(peak[j])
        count_of_peaks_threshold += 1
        location_peak.append(j)
        value_peak.append(peak[j])
    else:
        peak_threshold.append(0)

print "Peaks after applying threshold         :", count_of_peaks_threshold


valley = []
count_of_valleys = 0

location_valley = []


valley.append(0)
for j in range(1, no_frames - 1):
    if st_energy[j] < st_energy[j + 1] and st_energy[j] < st_energy[j - 1]:
        valley.append(st_energy[j])
        count_of_valleys += 1
        location_valley.append(j)
    else:
        valley.append(0)
valley.append(0)

print "Number of valleys                      :", count_of_valleys


location = location_peak + location_valley
location.sort()
ripple_valley = []
ripple_peak = []
ripple = []

for k in range(len(location_peak)):
    q = location.index(location_peak[k])
    ripple.append(location[q-1])
    ripple_valley.append(location[q-1])

    ripple.append(location[q])
    ripple_peak.append(location[q])

    ripple.append(location[q+1])
    ripple_valley.append(location[q+1])

value_valley = []
for j in range(len(ripple_valley)):
    value_valley.append(st_energy[ripple_valley[j]])

ripple_value = []
for k in range(1, len(ripple), 3):
    ripple_value.append((st_energy[ripple[k]]-st_energy[ripple[k+1]])/(st_energy[ripple[k]]-st_energy[ripple[k-1]]))
# print len(ripple_value)
# print len(location_peak)
ripple_value_thresh = []
c = 0
for k in range(len(ripple_value)):
    if ripple_value[k] > 0.5:
        ripple_value_thresh.append(ripple_value[k])
        c += 1
    else:
        ripple_value_thresh.append(0)
#

loc = []
peak_threshold[:] = []
for j in range(len(ripple_value_thresh)):
    if ripple_value_thresh[j] != 0:
        loc.append(location_peak[ripple_value.index(ripple_value_thresh[j])])
# print location_peak
print loc

for j in range(no_frames):

    if j in loc:
        peak_threshold.append(st_energy[loc.index(j)])
    else:
        peak_threshold.append(0)

print len(peak_threshold)



print "Vowel Approx: ", c


# for j in range(no_frames):
#     if ripple_value_thresh is not 0:
#         peak_threshold.append()
#     else:
#         peak_threshold.append(0)

#
# cv = 0
# len_rp = len(ripple_value)
# for j in range(len_rp):
#     if ripple_value[j] < 10:
#
#         ripple_value.pop(j)


# print "Count of vowels:", cv

# threshold = 0.1
#
# thresh = []
# for j in range(len(peak)):
#     thresh.append(threshold)
#
# count_of_peaks_threshold = 0
# peak_threshold = []
# for j in range(len(peak)):
#     if threshold < peak[j]:
#         peak_threshold.append(peak[j])
#         count_of_peaks_threshold += 1
#     else:
#         peak_threshold.append(0)


# print count_of_peaks_threshold
# plt.figure('Short Term Energy')
# plt.subplot(211)
# plt.plot(st_energy, 'black')
# plt.scatter(location_peak, value_peak, color = 'red')
# plt.scatter(ripple_valley, value_valley, color = 'green')
# for j in range(len(location_peak)):
#     plt.text(location_peak[j], value_peak[j], str(round(ripple_value[j],2)))
# plt.scatter(ripple, ripple_value, color='red')
# plt.stem(peak_threshold, 'red')
# plt.stem(valley, 'blue')
# plt.plot(thresh)
# plt.subplot(212)
# plt.plot(original_st_energy, 'red')
# plt.plot(thresh)
# plt.plot(original_st_energy, 'r')
# plt.stem(peak_threshold, 'black')
# plt.axis([0, len(st_energy), min(st_energy) + 0.1 * min(st_energy), max(st_energy) + 0.1 * max(st_energy)])
# plt.xlabel('No of frames')
# plt.ylabel('Normalised magnitude')
# plt.savefig(r'F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_5\ShortTermEnergy.png')

mark = []
for j in range(len(peak_threshold)):
    if peak_threshold[j] is not 0:
        mark.append(j * hop_size)
        mark.append(j * hop_size + window_size)


for i in range(0, len(mark), 2):
    text_file_4.write('%06.3f'%((mark[i] * 0.0000625)) + "\t" + '%06.3f'%((mark[i+1]*0.0000625)) + "\t" + "Vowel" + "\n")


# print threshold

# Plotting the sound waveform
# plt.figure('Sound Waveform')
# plt.plot(data)
# plt.axis([0, length, min(data) + 0.1 * min(data), max(data) + 0.1 * max(data)])
# plt.xlabel('No of samples')
# plt.ylabel('Normalised magnitude')
# plt.savefig(r'F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_5\SoundWaveform')

# plt.show()