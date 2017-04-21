#---------------------------------------------------------------------------------------------------------------------#
from scipy.io import wavfile
import numpy as np
import math
#---------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------#
audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\InputTestFile.wav'
window_dur = 30  # Duration of window in milliseconds
hop_dur = 10  # Hop duration in milliseconds
fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
data = data / float(2 ** 15) # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hamming(window_size)  # Window type: Hamming (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
noise_energy = 0
energy = [0] * length

#Squaring each data point
for j in range(length):
    energy[j] = data[j] * data[j]

#Calcilating noise energy
for j in range(0, 800):  # energy
    noise_energy += energy[j]
noise_energy /= 800
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
st_energy = [0] * no_frames
maximum = [0] * no_frames
frame_number = [0] * no_frames
start = [0] * no_frames
stop = [0] * no_frames

#Calculating frame wise short term energy
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    frame_number[i] = i
    start[i] = i * hop_size
    stop[i] = i * hop_size + window_size

    st_energy[i] = sum(frame ** 2)
#---------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------#
text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R1_Data_Value.txt', 'w')
for i in range(length):
    text_file_1.write(str(energy[i]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
text_file_2 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R2_Feature_file_frame.txt', 'w')
for i in range(no_frames):
    text_file_2.write("Frame No : " + str(i) + "\t" + "Short term energy : " + str(st_energy[i]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
#Finding the peaks
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

text_file_3 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R3_Peaks.txt', 'w')
for i in range(len(peak)):
    text_file_3.write("Frame No:" + str(i) + "\tValue:" + str(peak[i]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
#Finding the valleys
valley = []
count_of_valleys = 0

valley.append(0)
for j in range(1, no_frames - 1):
    if st_energy[j] < st_energy[j + 1] and st_energy[j] < st_energy[j - 1]:
        valley.append(st_energy[j])
        count_of_valleys += 1
    else:
        valley.append(0)
valley.append(0)

print "Number of valleys                      :", count_of_valleys

text_file_4 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R4_Valley.txt', 'w')
for i in range(len(valley)):
        text_file_4.write("Frame No:" + str(i) + "\tValue:" + str(valley[i]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
threshold = 0.04 * (noise_energy + (sum(peak) / count_of_peaks))
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
count_of_peaks_threshold = 0
peak_threshold = []
for j in range(len(peak)):
    if threshold < peak[j]:
        peak_threshold.append(peak[j])
        count_of_peaks_threshold += 1
    else:
        peak_threshold.append(0)

print "Peaks after applying threshold         :", count_of_peaks_threshold

peak1 = peak_threshold[:]

text_file_5 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R5_Peaks after applying threshold.txt', 'w')
for i in range(len(peak_threshold)):
        text_file_5.write("Frame No:" + str(i) + "\tValue:" + str(peak_threshold[i]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------#
count_of_peaks_threshold = 0
for j in range(len(peak_threshold) - 10):
    if peak_threshold[j] is not 0:
        for i in range(1, 10):
            if peak_threshold[j] < peak_threshold[j + i]:
                peak_threshold[j] = 0

for i in range(len(peak_threshold)):
    if peak_threshold[i] is not 0:
        count_of_peaks_threshold += 1
print "Peaks after removing adjacent maxima's :", count_of_peaks_threshold

peak2 = peak_threshold[:]

text_file_6 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R6_Peaks after removing adjacent maxima.txt', 'w')
for i in range(len(peak_threshold)):
        text_file_6.write("Frame No:" + str(i) + "\tValue:" + str(peak_threshold[i]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
consolidated_peak_and_valley = []
location = []
location_peak = []
location_valley = []

for j in range(len(peak_threshold)):
    if peak_threshold[j] is not 0:
        consolidated_peak_and_valley.append(peak_threshold[j])
        location.append(j)
        location_peak.append(j)
    if valley[j] is not 0:
        consolidated_peak_and_valley.append(valley[j])
        location.append(j)
        location_valley.append(j)

location_1 = location[:]
location_peak_1 = location_peak[:]
location_valley_1 = location_valley[:]

text_file_7 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R7_Peak Location.txt', 'w')
for j in range(len(location_peak_1)):
    text_file_7.write("Peak candidate no:" + str(j) + "\t" + "Location of peak:" + str(location_peak_1[j]) + "\tValue of peak:" + str(peak_threshold[location_peak_1[j]])+"\n")

text_file_8 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R8_Valley Location.txt', 'w')
for j in range(len(location_valley_1)):
    text_file_8.write("Valley no:\t" + str(j) + "\t\t" + "Location of valley:\t" + str(location_valley_1[j]) + "\n")

text_file_9 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R9_Consolidated Peak and Valley.txt', 'w')
for j in range(len(consolidated_peak_and_valley)):
    text_file_9.write("Location: " + str(location_1[j]) + "\t" + "Value:\t" + str(consolidated_peak_and_valley[j]))
    if location_1[j] in location_peak_1:
        text_file_9.write("\t\tPeak\n")
    else:
        text_file_9.write("\tValley\n")

#---------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------#
ripple = []
for k in range(len(location_peak)):
    q = location.index(location_peak[k])
    ripple.append(location[q-1])
    ripple.append(location[q])
    ripple.append(location[q+1])

ripple_1 = ripple[:]

text_file_10 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R10_Ripple.txt', 'w')
for j in range(len(ripple_1)):
    text_file_10.write(str(j) + "\t" + "Value:\t" + str(ripple_1[j]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------#
ripple_value = []
for k in range(1, len(ripple), 3):
    ripple_value.append((st_energy[ripple[k]]-st_energy[ripple[k+1]])/(st_energy[ripple[k]]-st_energy[ripple[k-1]]))

text_file_11 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R11_Ripple_Value.txt', 'w')
for j in range(len(ripple_value)):
    text_file_11.write(str(j) + "\t" + "Ripple(rq):\t" + str(ripple_value[j]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
for q in range(len(ripple_value)-1):
    if ((ripple_value[q] > 3) and (ripple_value[q+1] < 1.4)) or ((ripple_value[q] > 1.02) and (ripple_value[q+1] < 0.3)):
        if peak_threshold[location_peak[q]] > peak_threshold[location_peak[q+1]]:
            peak_threshold.pop(location_peak[q+1])
            peak_threshold.insert(location_peak[q+1], 0)

        if peak_threshold[location_peak[q]] < peak_threshold[location_peak[q + 1]]:
            peak_threshold.pop(location_peak[q])
            peak_threshold.insert(location_peak[q], 0)

peak3 = peak_threshold[:]


for j in range(len(peak_threshold)):
    if peak_threshold[j] is not 0:
        location_peak.append(j)


location_peak_cond1 = []

count_of_peaks_threshold = 0
for i in range(len(peak_threshold)):
    if peak_threshold[i] is not 0:
        count_of_peaks_threshold += 1
        location_peak_cond1.append(i)

# for j in range(len(location_peak_cond1)):
#     eliminate = location.index(location_peak_cond1[j])
#     location.pop(eliminate-1)
#     location.pop(eliminate)
#     location.pop(eliminate-1)

ripple[:] = []
for k in range(len(location_peak_cond1)):
    q = location.index(location_peak_cond1[k])
    ripple.append(location[q-1])
    ripple.append(location[q])
    ripple.append(location[q+1])

print "Peaks after first condition of ripple rejection :", count_of_peaks_threshold

text_file_12 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R12_Peaks Ripple Condition 1.txt', 'w')
for i in range(len(peak_threshold)):
        text_file_12.write("Frame No:" + str(i) + "\tValue:" + str(peak_threshold[i]) + "\n")

text_file_13 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R13_Updated Peak Location.txt', 'w')
for j in range(len(location_peak_cond1)):
    text_file_13.write("Peak candidate no:" + str(j) + "\t" + "Location of peak:" + str(location_peak_cond1[j]) + "\tValue of peak:" + str(peak_threshold[location_peak_cond1[j]])+"\n")

text_file_14 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R14_Updated Location.txt', 'w')
for j in range(len(location)):
    text_file_14.write("Location:\t" + str(location[j]) + "\n")
#---------------------------------------------------------------------------------------------------------------------#


peak_valley = []
note = []

for j in range(0, len(ripple), 1):
    peak_valley.append(st_energy[ripple[j]])

for j in range(1, len(ripple)-1, 3):
    if peak_valley[j]-peak_valley[j-1]/peak_valley[j] < 0.3 and peak_valley[j] - peak_valley[j+1]/peak_valley[j] <0.3 :
        note.append(j)

#---------------------------------------------------------------------------------------------------------------------#

text_file_16 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R16_Analysis.csv', 'w')
text_file_16.write("Frame No" + "," + "Peak" + "," + "Valley" + "," + "Peak 1" + "," + "Peak 2" + "," + "Peak 3" + "\n")
for j in range(len(peak_threshold)):
    text_file_16.write(str(j) + "," + str(peak[j]) + "," + str(valley[j]) + "," + str(peak1[j]) + "," + str(peak2[j]) + "," + str(peak_threshold[j]) + "\n")

#---------------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
mark = []
vowel_count = 0
for j in range(len(peak_threshold)):
    if peak_threshold[j] is not 0:
        mark.append(j * hop_size)
        mark.append(j * hop_size + window_size)
        vowel_count += 1


text_file_17 = open('F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\R17_Mark.csv', 'w')


for i in range(0, len(mark), 2):
    text_file_17.write(str(mark[i] * 0.0000625) + "\t" + str(mark[i+1]*0.0000625) + "\t" + str(i) + "\n")


print "Approx Vowel Count : ", vowel_count

work = [0] * length
for j in range(0, len(mark) - 1):
    work.insert(mark[j], 1)

#
# for i in range(0, len(mark), 2):
#     text_file_4.write(str(mark[i] * 0.0000625) + "\t" + str(mark[i+1]*0.0000625) + "\t" + str(i) + "\n")


go = [0] * length

for k in range(0, len(mark), 2):
    for j in range(mark[k], mark[k + 1]):
        go.pop(j)
        go.insert(j, 1)


data = data * go
#---------------------------------------------------------------------------------------------------------------------#

# Storing enhanced wavefile (Just to show input data array is written as output wav file)
wavfile.write("F:\Projects\Active Projects\Project Intern_IITB\Rough_Space_4\OutputTestFile1.wav", fs, data)
