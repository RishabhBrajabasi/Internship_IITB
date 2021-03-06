from scipy.io import wavfile
import numpy as np
import math
import matplotlib.pyplot as plt

audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\WorkPad\InputTestFile.wav'                # Provide the absolute path of wav file as argument to script  (input argument)
window_dur = 40                                                                                         # Duration of window in millisec
hop_dur    =  5                                                                                         # Hop duration in millisec

text_file_1   = open('F:\Projects\Active Projects\Project Intern_IITB\WorkPad\Feature_file_frame.csv', 'w')        # Output file for storing features
text_file_2   = open('F:\Projects\Active Projects\Project Intern_IITB\WorkPad\Feature_file_data.csv', 'w')         # Output file for storing features

fs, data = wavfile.read(audio_file)                                              # Reading data from wav file in an array
                                                                                 #  Returns Sample rate and data read from  wav file

data = data / (float(2 ** 15))                                                   # Normalizing it to [-1,1] range from [-2^15,2^15]

window_size = int(window_dur * fs * (0.001))                                     # Converting window length to samples
hop_size    = int(hop_dur * fs * (0.001))                                        # Converting hop length to samples
window_type = np.hamming(window_size)                                            # Window type: Hamming (by default)

no_frames   = int(math.ceil(len(data) / (float(hop_size))))                      # Determining the number of frames

zero_array  = np.zeros(window_size)                                              #Appending appropriate number of zeros
data        = np.concatenate((data, zero_array))

length      = len(data)                                                         #Finding length of the actual data

##### Plotting the sound waveform #####
plt.figure('Sound Waveform')
plt.plot(data)
plt.axis([0,length,min(data)+0.1*min(data),max(data)+0.1*max(data)])

# Printing Relevant Information
print " Window Size:",window_size,"samples"
print " Hop Size:",hop_size,"samples"
print " No of frames:",no_frames,"frames"
print " Length of data:",length,"samples"
print " Sampling Rate:",fs,"samples per second"

#Work on Data
energy = [0]*length

#Energy [Not really necessary to calculate]
for j in range(length):
    energy[j] = data[j]*data[j]


no_of_peak_in_sw=0
for j in range(1,len(data)-1):
    if(energy[j]>energy[j+1] and energy[j]>energy[j-1]):
        no_of_peak_in_sw = no_of_peak_in_sw + 1

print "No of peaks in data:",no_of_peak_in_sw
################################## ?????????????????????????? ########################
                                    ######HOW MANY SAMPLES #####

#Calculating Noise Energy for use in thresholding operations
noise_energy=0
for j in range(0,800):       #energy
    noise_energy = noise_energy + data[j]
noise_energy=noise_energy/800

################################## ?????????????????????????? ########################

#Work on Frames
st_energy = [0] * no_frames
maximum = [0] * no_frames

#fileNameTemplate = r'C:\Users\Mahe\Desktop\Hopethisworks\Plot{0:02d}.png'

for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    # Do frame wise processing here

      # ---------- For example: Computing short time energy -------------------
    st_energy[i] = sum(frame ** 2)
    maximum[i] = max(frame)


    ## Code to individually plot each of the frames created. Since, 4097 plots will be created, run it sparingly
    # plt.plot(frame)
    # plt.savefig(fileNameTemplate.format(i), format='png')
    # plt.clf()

    # ------------------------------------------------------------------------

    # ------------------------------

count_of_peaks   = 0
count_of_valleys = 0

#Find the peaks[Maxima]
peak=[]
for j in range(1,no_frames-1):
    if(st_energy[j]>st_energy[j+1] and st_energy[j]>st_energy[j-1]):
        peak.append(st_energy[j])
        count_of_peaks = count_of_peaks + 1
    else:
        peak.append(0)

#Find the valleys[Minima]
valley=[]
for j in range(1,no_frames-1):
    if(st_energy[j]<st_energy[j+1] and st_energy[j]<st_energy[j-1]):
        valley.append(st_energy[j])
        count_of_valleys = count_of_valleys +1
    else:
        valley.append(0)

#Threshold operation to eliminate peaks
threshold = (0.04)* (noise_energy + (sum(peak)/count_of_peaks))
print " Threshold",threshold

#Calculating peaks which are greater than the threshold
count_of_maxi = 0
maxi=[]

for j in range(len(peak)):
    if (threshold < peak[j]):
        maxi.append(peak[j])
        count_of_maxi = count_of_maxi + 1
    else:
        maxi.append(0)

# Removing adjacent peaks
for j in range(len(maxi)-100):
    if(maxi[j] is not 0):
        for i in range(1,100):
            if(maxi[j]<maxi[j+i]):
                maxi[j]=0


print "count_of_peaks",count_of_peaks
print "count_of_valleys",count_of_valleys
print "Peaks after thresholding ", count_of_maxi

# Storing extracted features in file for Frame
# text_file_1.write("Frames" + "," + "Short term energy" + "," + "Maximum" + "\n")
# for i in range(no_frames):
#     text_file_1.write(str(i + 1) + "," + str(st_energy[i]) + "," + str(maximum[i]) + "\n")

# Storing extracted features in file for Data
# text_file_2.write("Data"+"\n")
# for i in range(length):
#     text_file_2.write(str(energy[i]) + "\n")

#plotting the data
plt.figure('Short term energy of frames')
plt.subplot(211)
plt.plot(st_energy,'g')
plt.xscale(0,length/16000)
plt.subplot(212)
plt.plot(data)
plt.title('Short term energy of frames')

plt.figure('Peaks & Valleys')
plt.subplot(211)
plt.stem(peak,'r')
plt.title('Peaks')

plt.subplot(212)
plt.stem(valley,'b')
plt.title('Valley')

plt.figure('Summary')
plt.subplot(111)
plt.plot(st_energy)
plt.subplot(111)
plt.plot(maxi)
plt.stem(maxi,'y')

plt.show()

# Storing enhanced wavefile (Just to show input data array is written as output wav file)
#wavfile.write("F:\Projects\Active Projects\Project Intern_IITB\WorkPad\OutputTestFile.wav", fs, data)



