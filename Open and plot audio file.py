import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import csv

#Reading the audio file(Path specification, Find default location
audio_file = wave.open('C:\Users\Mahe\Desktop\Set 1\ABX_D_27122016_1_1.wav','r')
#Reading the values and storing in csv
fl = open('C:\Users\Mahe\Desktop\udio_data.csv', 'w')

channels=audio_file.getnchannels()
print "No of channels:",channels
samplewidth=audio_file.getsampwidth()
print "Sample Width(bytes):",samplewidth
samplingfreq=audio_file.getframerate()
print "Sampling Frequency:",samplingfreq
compression=audio_file.getcomptype()
print "Compression:",compression

#Extract Raw Audio from Wav File
#-1 returns the full audio sample . total length is 655600
#any integer n returns those many frames
signal = audio_file.readframes(-1)
signal = np.fromstring(signal, 'Int16')

#print len(signal)
#framelength=len(signal)
#frame = audio_file.readframes(framelength/1000)
#frame = np.fromstring(frame, 'Int16')

#If Stereo
if audio_file.getnchannels() == 2:
    print 'Just mono files'
    sys.exit(0)

writer = csv.writer(fl)
#writer.writerow(['label1', 'label2', 'label3']) #if needed
#for values in signal:
writer.writerow(signal)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()

#plt.figure(2)
#plt.title('FRAME...')
#plt.plot(frame)
#plt.show()

fl.close()