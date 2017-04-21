from __future__ import division

import math
import matplotlib.pyplot as plt
import numpy
from scipy.io import wavfile

def moving_average(interval, window_size):
    window = numpy.ones(int(window_size)) / float(window_size)
    return numpy.convolve(interval, window, 'same')


audiofile ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_Repeat_2\\211.wav'
window_dur = 50
hop_dur = 7
threshold_smooth =100
fs, data = wavfile.read(audiofile)
data = data / float(2 ** 15)
window_size = int(window_dur * fs * 0.001)
hop_size = int(hop_dur * fs * 0.001)
window_type = numpy.hanning(window_size)
no_frames = int(math.ceil(len(data) / (float(hop_size))))
zero_array = numpy.zeros(window_size)
data = numpy.concatenate((data, zero_array))
length = len(data)
x_values = numpy.arange(0, len(data), 1) / float(fs)

def shortTermEnergy(frame):
  return sum( [ abs(x)**2 for x in frame ] ) / len(frame)

def zeroCrossingRate(frame):
  signs             = numpy.sign(frame)
  signs[signs == 0] = -1

  return len(numpy.where(numpy.diff(signs))[0])/len(frame)

def chunks(l, k):
  for i in range(0, len(l), k):
    yield l[i:i+k]

def entropyOfEnergy(frame, numSubFrames):
  lenSubFrame = int(numpy.floor(len(frame) / numSubFrames))
  shortFrames = list(chunks(frame, lenSubFrame))
  energy      = [ shortTermEnergy(s) for s in shortFrames ]
  totalEnergy = sum(energy)
  energy      = [ e / totalEnergy for e in energy ]

  entropy = 0.0
  for e in energy:
    if e != 0:
      entropy = entropy - e * numpy.log2(e)

  return entropy

st_energy = []
entropy = []
zcr = []
for i in range(no_frames):  # Calculating frame wise short term energy
    frame = data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hamming window
    entropy.append(entropyOfEnergy(frame,5))
    st_energy.append(shortTermEnergy(frame))
    zcr.append(zeroCrossingRate(frame))

plt.subplot(311)
plt.plot(x_values,data)
plt.xlim(0,x_values[-1])
plt.title('Sound Waveform')
#
# plt.subplot(412)
# plt.plot(moving_average(entropy,5))
# plt.xlim(0,len(entropy))

plt.subplot(312)
plt.plot(moving_average(st_energy,5))
plt.xlim(0,len(st_energy))
plt.title('Short Term Energy')

plt.subplot(313)
plt.plot(moving_average(zcr,5))
plt.xlim(0,len(zcr))
plt.title('Zero Crossing Rate')

plt.show()