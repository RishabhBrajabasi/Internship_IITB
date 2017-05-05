from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

audio_file ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Analyze\Vowel_Evaluation_V4_I2\\17.wav'


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

data2 = []
for j in range(len(data)):
    if data[j] < 0:
        data2.append(data[j]*-0)
    else:
        data2.append(data[j])

max1 = max(data1)
max2 = max(data2)
for k in range(len(data1)):
    data1[k] = data1[k]/max1
    data2[k] = data2[k]/max2

d1 = moving_average(data1, 1000)
d1 = butter_bandpass_filter(d1, 100, fs, 5)
d1 = moving_average(d1, 20)

d2 = moving_average(data2, 1000)
d2 = butter_bandpass_filter(d2, 100, fs, 5)
d2 = moving_average(d2, 20)

max1 = max(d1)
for k in range(len(d1)):
    d1[k] = d1[k]/max1
max2 = max(d2)
for k in range(len(d2)):
    d2[k] = d2[k]/max2

plt.plot(x_values, batman, 'blue', label='Audio')
plt.plot(x_values, d1, 'red', label='Envelope', linewidth='2.0')
plt.legend(loc='best')
plt.show()
d1 = d1.tolist()
convex_hull = []
segment_boundary = [0]

def segmentation(frame_energy, original_curve):
    threshold = 0.1
    convex_hull[:] = []  # The list needs to be emptied for each iteration
    break_point = frame_energy.index(max(
        frame_energy))  # The point till which the convex hull is monotonically increasing and following which it is monotonically decreasing
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
            st = original_curve.index(frame_energy[0])  # Starting index of segment 1
            bp = original_curve.index((frame_energy[diff_f.index(max(diff_f))]))  # Ending index of segment 1, and starting index of segment 2
            sp = original_curve.index(frame_energy[-1])  # Ending index of segment 2
            segment_boundary.append(bp)  # Adding the breakpoint to the segment boundary list
            if len(original_curve[st:bp]) > 0 and len(original_curve[bp:sp]) > 0:  # If both segments are larger than 0 in length, then proceed with further segmentation
                segmentation(original_curve[st:bp], original_curve)
                segmentation(original_curve[bp:sp], original_curve)
            else:
                return ()
        else:
            return ()
    else:
        if max(diff_b) > threshold:
            st = original_curve.index(frame_energy[0])
            bp = original_curve.index((frame_energy[diff_b.index(max(diff_b))]))
            sp = original_curve.index(frame_energy[-1])
            segment_boundary.append(bp)
            if len(original_curve[st:bp]) > 0 and len(original_curve[bp:sp]) > 0:
                segmentation(original_curve[st:bp], original_curve)
                segmentation(original_curve[bp:sp], original_curve)
            else:
                return ()
        else:
            return ()

segmentation(d1, d1)

segment_boundary.append(len(d1))
segment_boundary.sort()

plt.plot(batman, 'blue', label='Audio')
plt.plot(d1, 'red', label='Envelope', linewidth='2.0')
plt.vlines(segment_boundary[0], min(d1), max(d1), 'black', label='Segment boundary', linewidth='2.0')
for j in segment_boundary:
    plt.vlines(j, min(d1), max(d1), 'black')
plt.ylabel('Magnitude')
plt.xlabel('Frame Number')
plt.title('Segments')
plt.legend(loc='best')
plt.show()
