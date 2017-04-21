from scipy.io import wavfile
import math
import numpy as np
import matplotlib.pyplot as plt


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
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


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


# audiofile = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Data 1\\00001laxmi_56580b937e63f5035c003431_57b19ede9ee20a03d87b5f89_19_4000211.wav'
audiofile = 'C:\Users\Mahe\Desktop\\17.wav'
# audiofile ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Analyze\Vowel_Evaluation_V4_I2\\17.wav'

fs, data = wavfile.read(audiofile)  # Reading data from wav file in an array
window_dur = 50
hop_dur = 7
data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data
x_values = np.arange(0, len(data), 1) / float(fs)

#######################################################################################################################
# Plotting the original sound waveform
plt.figure('Sound Waveform')
plt.plot(x_values, data)
plt.xlabel('Time in seconds')
plt.ylabel('Amplitude')
plt.title('Sound Waveform')
plt.show()
#######################################################################################################################


#######################################################################################################################
st_energy_square = []
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    st_energy_square.append(sum(frame ** 2))
norm_max_square = max(st_energy_square)
print 'SQUARE',type(st_energy_square)

st_energy_rms = []
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    st_energy_rms.append(math.sqrt(sum(frame ** 2)/len(frame)))
norm_max_rms = max(st_energy_rms)
print 'RMS',type(st_energy_rms)

st_energy_absolute = []
for i in range(no_frames):
    frame = np.abs(data[i * hop_size:i * hop_size + window_size] * window_type)
    st_energy_absolute.append(sum(frame))
norm_max_absolute = max(st_energy_absolute)

st_energy_teager = []
for i in range(no_frames):
    frame = np.abs(data[i * hop_size:i * hop_size + window_size] * window_type)
    for j in range(1,len(frame)-1,1):
        frame[j] = frame[j]*frame[j] - frame[j+1]*frame[j-1]
    st_energy_teager.append(sum(frame))
norm_max_teager = max(st_energy_teager)

for i in range(no_frames):
    st_energy_square[i] = st_energy_square[i]/norm_max_square
    st_energy_rms[i] = st_energy_rms[i]/norm_max_rms
    st_energy_teager[i] = st_energy_teager[i]/norm_max_teager
    st_energy_absolute[i] = st_energy_absolute[i]/norm_max_absolute

plt.figure('Original Short Term Energy')
plt.plot(st_energy_square)
plt.ylabel('Magnitude')
plt.xlabel('Frame Number')
plt.title('Short Term Energy [SQUARE]')
plt.show()

plt.figure('Original Short Term Energy')
plt.plot(st_energy_absolute)
plt.ylabel('Magnitude')
plt.xlabel('Frame Number')
plt.title('Short Term Energy[ABSOLUTE]')
plt.show()

plt.figure('Original Short Term Energy')
plt.plot(st_energy_rms)
plt.ylabel('Magnitude')
plt.xlabel('Frame Number')
plt.title('Short Term Energy[RMS]')
plt.show()

plt.figure('Original Short Term Energy')
plt.plot(st_energy_teager)
plt.ylabel('Magnitude')
plt.xlabel('Frame Number')
plt.title('Short Term Energy[TEAGER]')
plt.show()

plt.figure('All plots')

plt.subplot(221)
plt.plot(st_energy_rms,label='RMS',color='red')
plt.title('RMS')
plt.xlabel('No of frames')
plt.ylabel('Normalised Magnitude')

plt.subplot(222)
plt.plot(st_energy_absolute,label='ABSOLUTE',color='blue')
plt.title('ABSOLUTE')
plt.xlabel('No of frames')
plt.ylabel('Normalised Magnitude')

plt.subplot(223)
plt.plot(st_energy_square,label='SQUARE',color='green')
plt.title('SQUARE')
plt.xlabel('No of frames')
plt.ylabel('Normalised Magnitude')
plt.subplot(224)
plt.plot(st_energy_teager,label='TEAGER',color='black')
plt.title('TEAGER')
plt.xlabel('No of frames')
plt.ylabel('Normalised Magnitude')
plt.show()

plt.figure('All plots')
plt.plot(st_energy_rms,label='RMS',color='red')
plt.plot(st_energy_absolute,label='ABSOLUTE',color='blue')
plt.plot(st_energy_square,label='SQUARE',color='green')
plt.plot(st_energy_teager,label='TEAGER',color='black')
plt.xlabel('No of frames')
plt.ylabel('Normalised Magnitude')
plt.legend()
plt.show()

plt.figure('All plots')
plt.subplot(511)
plt.plot(x_values,data)
plt.xlabel('Time in seconds')
plt.ylabel('Amplitude')


plt.subplot(512)
plt.plot(st_energy_rms,label='RMS',color='red')
plt.legend()
plt.xlabel('No of frames')
plt.ylabel('NM')

plt.subplot(513)
plt.plot(st_energy_absolute,label='ABSOLUTE',color='blue')
plt.legend()
plt.xlabel('No of frames')
plt.ylabel('NM')

plt.subplot(514)
plt.plot(st_energy_square,label='SQUARE',color='green')
plt.legend()
plt.xlabel('No of frames')
plt.ylabel('NM')

plt.subplot(515)
plt.plot(st_energy_teager,label='TEAGER',color='black')
plt.legend()
plt.xlabel('No of frames')
plt.ylabel('NM')

plt.show()