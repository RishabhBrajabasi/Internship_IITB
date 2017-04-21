"""
Segmentation.py
Aim : 1. To find/detect the syllabic units in the speech signal
      2. To compare output against force aligned output.

Computing the short term energy of the speech signal.
Applying the Convex hull algorithm on the short term energy curve to obtain the syllabic units.
Compare output obtained with force aligned text.

Script runs on individual files.
Path of all files related to code.
F:\Projects\Active Projects\Project Intern_IITB\Segmentation

Date: 18th April 2017
Author: Rishabh Brajabasi
"""
from scipy.io import wavfile
import math
from shutil import copyfile
import numpy as np
import csv
import matplotlib.pyplot as plt

def moving_average(interval, window):
    window = np.ones(int(window)) / float(window)
    return np.convolve(interval, window, 'same')

def segmentation(frame_energy):
    threshold = 0.2  # The difference between the convex hull and the frame_energy
    convex_hull[:] = []  # The list needs to be emptied for each iteration
    break_point = frame_energy.index(max(frame_energy))  # The point till which the convex hull is monotonically increasing and following which it is monotonically decreasing
    last_1 = frame_energy[0]  # The value that the convex hull sticks to if the curve is not increasing. Updated later
    last_2 = frame_energy[break_point]  # The value the convex hull sticks to if the curve is not decreasing. Updated Later

    for k in range(len(frame_energy) - 1):
        if k < break_point:  # Monotonically increase till breakpoint reached
            if frame_energy[k] > frame_energy[k - 1] and frame_energy[k] > last_1:
                convex_hull.append(frame_energy[k])  # Add element of the frame_energy to the convex hull as long as it is increasing
                last_1 = frame_energy[k]  # Update last_1, so that if the curve starts decreasing, the last greatest value is stored in last_1
            else:
                convex_hull.append(last_1)  # If the frame_energy curve starts decreasing, the last highest value is assigned to the convex hull

        elif k > break_point:  # Monotonically decreasing from breakpoint till end of the segment
            if frame_energy[k + 1] < frame_energy[k] < last_2:
                convex_hull.append(frame_energy[k])  # Add element of the frame_energy to the convex hull as long as it is decreasing
                last_2 = frame_energy[k + 1]  # Update last_2, so that if the curve starts increasing, the last smallest value is stored in last_2
            else:
                convex_hull.append(last_2)  # If the frame_energy curve starts increasing, the last lowest value is assigned to the convex hull

        else:
            convex_hull.append(frame_energy[break_point])  # The point of inflection

    convex_hull.append(frame_energy[len(frame_energy) - 1])  # Had to compute one less point due to indexing issues. Appending the last missing point with last least value

    diff_f = [0] * len(convex_hull)  # Creating a list of the same length as the Convex hull and assigning all its elements with a value of zero
    for k in range(len(convex_hull)):
        diff_f[k] = convex_hull[k] - frame_energy[k]  # Finding the difference between the convex hull and the frame_energy
        if diff_f[k] < 0:
            diff_f[k] *= 0  # For those points where the frame_energy is greater than the convex hull, making that point 0.

    frame_energy.reverse()  # Reverse the frame_energy and run convex hull on it again.

    convex_hull[:] = []  # Emptying the list
    break_point = frame_energy.index(max(frame_energy))
    last_1 = frame_energy[0]
    last_2 = frame_energy[break_point]

    for k in range(len(frame_energy) - 1):
        if k < break_point:
            if frame_energy[k] > frame_energy[k - 1] and frame_energy[k] > last_1:
                convex_hull.append(frame_energy[k])
                last_1 = frame_energy[k]
            else:
                convex_hull.append(last_1)

        elif k > break_point:
            if frame_energy[k + 1] < frame_energy[k] < last_2:
                convex_hull.append(frame_energy[k])
                last_2 = frame_energy[k + 1]
            else:
                convex_hull.append(last_2)

        else:
            convex_hull.append(frame_energy[break_point])

    convex_hull.append(frame_energy[len(frame_energy) - 1])
    diff_b = [0] * len(convex_hull)
    for k in range(len(convex_hull)):
        diff_b[k] = convex_hull[k] - frame_energy[k]
        if diff_b[k] < 0:
            diff_b[k] *= 0

    frame_energy.reverse()  # Reverse frame_energy once more for correct indexing in proceeding steps
    diff_b.reverse()  # Reverse diff_b for correct indexing in proceeding steps

    if max(diff_f) >= max(diff_b):  # Comparing Maximum's
        if max(diff_f) > threshold:  # Comparing the diff with a unified threshold which decides whether the difference is large enough or not to warrant a segment
            st = st_energy.index(frame_energy[0])  # Starting index of segment 1
            bp = st_energy.index((frame_energy[diff_f.index(max(diff_f))]))  # Ending index of segment 1, and starting index of segment 2
            sp = st_energy.index(frame_energy[-1])  # Ending index of segment 2
            segment_boundary.append(bp)  # Adding the breakpoint to the segment boundary list
            if len(st_energy[st:bp]) > 0 and len(st_energy[bp:sp]) > 0:  # If both segments are larger than 0 in length, then proceed with further segmentation
                segmentation(st_energy[st:bp])
                segmentation(st_energy[bp:sp])
            else:
                return ()
        else:
            return ()
    else:
        if max(diff_b) > threshold:
            st = st_energy.index(frame_energy[0])
            bp = st_energy.index((frame_energy[diff_b.index(max(diff_b))]))
            sp = st_energy.index(frame_energy[-1])
            segment_boundary.append(bp)
            if len(st_energy[st:bp]) > 0 and len(st_energy[bp:sp]) > 0:
                segmentation(st_energy[st:bp])
                segmentation(st_energy[bp:sp])
            else:
                return ()
        else:
            return ()

def text_grid(csv_file_name):
    csvFileName = csv_file_name
    TGFileName = csvFileName.split('.')[0] + '.TextGrid'  # Setting name of TextGrid file

    fidcsv = open(csvFileName, 'r')
    fidTG = open(TGFileName, 'w')

    reader = csv.reader(fidcsv, delimiter="\t")  # Reading data from csv file
    data_1 = list(reader)  # Converting read data into python list format
    label_count = len(data_1)  # Finding total number of rows in csv file
    end_time = data_1[-1][1]


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

    for k in range(label_count):
        fidTG.write('\t\tintervals [' + str(k) + ']:\n')
        fidTG.write('\t\t\txmin = ' + str(data_1[k][0]) + '\n')
        fidTG.write('\t\t\txmax = ' + str(data_1[k][1]) + '\n')
        fidTG.write('\t\t\ttext = "' + str(data_1[k][2]) + '"\n')

    fidcsv.close()
    fidTG.close()
    return ()

audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\Analyze\Vowel_Evaluation_V6_Test_7\\17.wav'
csv_file = 'F:\Projects\Active Projects\Project Intern_IITB\Segmentation\Segments.csv'
copyfile(audio_file,'F:\Projects\Active Projects\Project Intern_IITB\Segmentation\Audio.wav')
copyfile(audio_file[:-4] + 'FA.TextGrid','F:\Projects\Active Projects\Project Intern_IITB\Segmentation\Force Aligned Transcription.TextGrid')

window_dur = 30  # Duration of window in milliseconds
hop_dur = 5  # Hop duration in milliseconds
fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
window_type = np.hanning(window_size)  # Window type: Hanning (by default)
no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
data = np.concatenate((data, zero_array))
length = len(data)  # Finding length of the actual data

st_energy = []
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    st_energy.append(sum(frame ** 2))

max_st_energy = max(st_energy)  # Maximum value of Short term energy curve
for i in range(no_frames):
    st_energy[i] = st_energy[i] / max_st_energy  # Normalizing the curve


st_energy = moving_average(st_energy,10)
st_energy = st_energy.tolist()

convex_hull = []
segment_boundary = [0]
segmentation(st_energy)
segment_boundary.append(len(st_energy))
segment_boundary.sort()

text_file_1 = open(csv_file, 'w')

mark = []
for j in segment_boundary:
    mark.append(j * hop_size)

for i in range(0, len(mark)-1):
    text_file_1.write('%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i+1] * 0.0000625) + "\t" + "Segment " + str(i+1) + "\n")
text_file_1.close()

text_grid(csv_file)


plt.plot(st_energy, 'blue', label='Short term energy')
plt.vlines(segment_boundary[0], min(st_energy), max(st_energy), 'black', label='Segment boundary')
for j in segment_boundary:
    plt.vlines(j, min(st_energy), max(st_energy), 'black')
plt.ylabel('Magnitude')
plt.xlabel('Frame Number')
plt.title('Segments')
plt.legend()
plt.show()