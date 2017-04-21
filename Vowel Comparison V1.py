from scipy.io import wavfile
import math
import numpy as np
import glob
import re
from datetime import datetime
import winsound
import csv


startTime = datetime.now()  # To calculate the run time of the code.

results_vowel = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Vowel_count_V28.csv', 'a')  # The csv file where the results are saved

results_vowel.write('Name' + ',' + 'Peak_Elimination' + ',' + 'Evaluation Script PE' + ',' + 'Convex Hull' + ',' + 'Evaluation Script CH' + ',' + 'Vowel Count FA' + ',' + 'Duration(In Seconds)' '\n')    # Column headers


# Smoothing algorithm
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as npy
    from math import factorial
    try:
        window_size = npy.abs(npy.int(window_size))
        order = npy.abs(npy.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


#  Algorithm no 1 to detect vowel centres
def peak_elimination(audiofile):
    try:
        audio_file = audiofile
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

        noise_energy = 0
        energy = [0] * length
        for bit in range(length):
            energy[bit] = data[bit] * data[bit]  # Squaring each point of the data to calculate noise energy
        for ne in range(0, 800):
            noise_energy += energy[ne]  # Taking the first 800 samples of the original sound file
        noise_energy /= 800  # Averaging the square of the first 800 noise samples

        st_energy = [0] * no_frames
        for i in range(no_frames):  # Calculating frame wise short term energy
            frame = data[i * hop_size:i * hop_size + window_size] * window_type  # Multiplying each frame with a hanning window
            st_energy[i] = sum(frame ** 2)  # Calculating the short term energy
        st_energy = savitzky_golay(st_energy, 51, 3)  # window size 51, polynomial order 3

        peak = []
        count_of_peaks = 0
        for p in range(len(st_energy)):
            if p == 0:  # First element
                if st_energy[p] > st_energy[p+1]:  # If the first element is greater than the succeeding element it is a peak.
                    peak.append(st_energy[p])  # Append the energy level of the peak
                    count_of_peaks += 1  # Increment count
                else:
                    peak.append(0)  # Else append a zero
            elif p == len(st_energy) - 1:  # Last element
                if st_energy[p] > st_energy[p-1]:  # If the last element is greater than the preceding element it is a peak.
                    peak.append(st_energy[p])  # Append the energy level of the peak
                    count_of_peaks += 1  # Increment count
                else:
                    peak.append(0)  # Else append a zero
            else:  # All the other elements
                if st_energy[p] > st_energy[p + 1] and st_energy[p] > st_energy[p - 1]:  # If the element is greater than the element preceding and succeeding it, it is a peak.
                    peak.append(st_energy[p])  # Append the energy level of the peak
                    count_of_peaks += 1  # Increment count
                else:
                    peak.append(0)  # Else append a zero

        threshold = 0.01 + 0.04 * (noise_energy + (sum(peak) / count_of_peaks))  # The threshold which eliminates minor peaks.

        count_of_peaks_threshold = 0
        peak_threshold = []
        location_peak = []
        for p in range(len(peak)):
            if threshold < peak[p]:  # If the peak value is greater than the threshold
                peak_threshold.append(peak[p])  # Append the energy level to a new list
                count_of_peaks_threshold += 1  # Increment count
                location_peak.append(p)  # Make note of the location of the peak
            else:
                peak_threshold.append(0)  # Else append zero

        # What : Finding the valleys on the short term energy curve
        valley = []
        count_of_valleys = 0
        location_valley = []
        for p in range(len(st_energy)):
            if p == 0:  # For the first element
                if st_energy[p] < st_energy[p + 1]:  # If the first element is lesser than the succeeding element
                    valley.append(st_energy[p])  # Append the energy level of the valley
                    count_of_valleys += 1  # Increment the count
                    location_valley.append(p)  # Make note of the position of the valley
                else:
                    valley.append(0)  # Else append zero
            elif p == len(st_energy) - 1:  # For the last element
                if st_energy[p] < st_energy[p-1]:  # If the last element is lesser than the preceding element
                    valley.append(st_energy[p])  # Append the energy level of the valley
                    count_of_valleys += 1  # Increment the count
                    location_valley.append(p)  # Make note of the position of the valley
                else:
                    valley.append(0)  # Else append zero
            else:
                if st_energy[p] < st_energy[p + 1] and st_energy[p] < st_energy[p - 1]:  # If the element is lesser than the element preceding and succeeding it
                    valley.append(st_energy[p])  # Append the energy level of the valley
                    count_of_valleys += 1  # Increment the count
                    location_valley.append(p)  # Make note of the position of the valley
                else:
                    valley.append(0)  # Else append zero

        location = location_peak + location_valley  # Combing the list of the location of all the peaks and valleys
        location.sort()  # Sorting it so that each peak has a valley to it's left and right
        ripple_valley = []
        ripple_peak = []
        ripple = []

        # What we need is only the valleys to the left and right of the peak. The other valleys are not important
        for k in range(len(location_peak)):
            q = location.index(location_peak[k])  # Extracting the location of the peak
            if location_peak[k] == len(peak) - 1:  # If the peak is the last element of the short term energy curve
                ripple.append(location[q - 1])  # The location of the valley before the last peak is added
                ripple_valley.append(location[q - 1])  # The location of the valley before the last peak is added
                ripple.append(location[q])  # The location of the peak is added
                ripple_peak.append(location[q])  # The location of the peak is added
                ripple.append(location[q - 1])  # The location of the valley before the last peak is added, as there is no valley after it
                ripple_valley.append(location[q - 1])  # The location of the valley before the last peak is added, as there is no valley after it
            elif location_peak[k] == 0:  # If the peak is the first element of the short term energy curve
                ripple.append(location[q + 1])  # The location of the valley after the first peak is added
                ripple_valley.append(location[q + 1])  # The location of the valley after the first peak is added
                ripple.append(location[q])  # The location of the peak is added
                ripple_peak.append(location[q])  # The location of the peak is added
                ripple.append(location[q + 1])  # The location of the valley after the first peak is added, as there is no valley after it
                ripple_valley.append(location[q + 1])  # The location of the valley after the first peak is added, as there is no valley after it
            else:  # For every other element
                ripple.append(location[q - 1])  # The location of the valley before the peak is added
                ripple_valley.append(location[q - 1])  # The location of the valley before the peak is added
                ripple.append(location[q])  # The location of the peak is added
                ripple_peak.append(location[q])  # The location of the peak is added
                ripple.append(location[q + 1])  # The location of the valley after the peak is added
                ripple_valley.append(location[q + 1])  # The location of the valley after the peak is added

        value_valley = []
        for p in range(len(ripple_valley)):
            value_valley.append(st_energy[ripple_valley[p]])  # Get the energy level for each of the relevant valleys

        ripple_value = []
        for k in range(1, len(ripple), 3):
            ripple_value.append(
                (st_energy[ripple[k]] - st_energy[ripple[k + 1]]) / (st_energy[ripple[k]] - st_energy[ripple[k - 1]]))  # Calculating the ripple value for each peak

        # What : Shortlist peaks based on their ripple value.
        ripple_value_thresh = []
        count_of_vowels = 0
        for k in range(len(ripple_value)):
            if 2.0 >= ripple_value[k] >= 0.4:  # If ripple value is within the specified range, the peak is determined to be a vowel centre
                ripple_value_thresh.append(ripple_value[k])
                count_of_vowels += 1  # Increment count of vowel
            else:
                ripple_value_thresh.append(0)

        results_vowel.write(audiofile + ',' + str(count_of_vowels))  # Writing the results to the main results file

        loc = []
        peak_threshold[:] = []
        for p in range(len(ripple_value_thresh)):  # Finding the location of the peaks that have been chosen
            if ripple_value_thresh[p] != 0:
                loc.append(location_peak[ripple_value.index(ripple_value_thresh[p])])

        for p in range(no_frames):  # Finding the energy level of the peaks that have been chosen
            if p in loc:
                peak_threshold.append(st_energy[loc.index(p)])
            else:
                peak_threshold.append(0)

        text_file_1 = open(audiofile[:-4] + 'PE.csv', 'w')  # Opening CSV file to store results and to create TextGrid

        mark = []
        for p in range(len(peak_threshold)):  # Extracting a 50 ms slice of the audio file based on the frame number
            if peak_threshold[p] is not 0:
                mark.append(p * hop_size)
                mark.append(p * hop_size + window_size)

        for i in range(0, len(mark), 2):  # Writing the result to a CSV File
            text_file_1.write(
                '%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + "Vowel" + "\n")

    except:
        results_vowel.write(audiofile + ',' + 'Error')


#  Algorithm no 2 to detect vowel centres
def convexhull(audiofile):
    try:
        audio_file = audiofile
        window_dur = 50  # Duration of window in milliseconds
        hop_dur = 5  # Hop duration in milliseconds
        fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
        data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
        window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
        hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
        window_type = np.hanning(window_size)  # Window type: Hanning (by default)
        no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
        zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
        data = np.concatenate((data, zero_array))

        st_energy = [0] * no_frames
        for i in range(no_frames):  # Calculating frame wise short term energy
            frame = data[i * hop_size:i * hop_size + window_size] * window_type
            st_energy[i] = sum(frame ** 2)

        st_energy = savitzky_golay(st_energy, 51, 3)  # window size 51, polynomial order 3
        st_energy = st_energy.tolist()

        convex_hull = []
        segment_boundary = [0]

        def segmentation(frame_energy):
            threshold = 0.3  # The difference between the convex hull and the frame_energy
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
                        segmentation(st_energy[st:bp])  # Run it on the first segment
                        segmentation(st_energy[bp:sp])  # Run it on the second segment
                    else:
                        return ()  # Return
                else:
                    return ()  # Return
            else:
                if max(diff_b) > threshold:  # Comparing the diff with a unified threshold which decides whether the difference is large enough or not to warrant a segment
                    st = st_energy.index(frame_energy[0])  # Starting index of segment 1
                    bp = st_energy.index((frame_energy[diff_b.index(max(diff_b))]))  # Ending index of segment 1, and starting index of segment 2
                    sp = st_energy.index(frame_energy[-1])  # Ending index of segment 2
                    segment_boundary.append(bp)  # Adding the breakpoint to the segment boundary list
                    if len(st_energy[st:bp]) > 0 and len(st_energy[bp:sp]) > 0:  # If both segments are larger than 0 in length, then proceed with further segmentation
                        segmentation(st_energy[st:bp])  # Run it on the first segment
                        segmentation(st_energy[bp:sp])  # Run it on the second segment
                    else:
                        return ()  # Return
                else:
                    return ()  # Return

        segmentation(st_energy)

        # What : Since the boundaries are not determined in a sequential order, they need to be sorted before we can find the peaks in each segment
        segment_boundary.append(len(st_energy))
        segment_boundary.sort()

        # What : In each of the segments determined by the convex hull algorithm, we find the peak and mark that as the vowel centre.
        peaks = []
        for seg in range(0, len(segment_boundary) - 1):
            if len(st_energy[segment_boundary[seg]:segment_boundary[seg + 1]]) != 0:
                peaks.append(st_energy.index(max(st_energy[segment_boundary[seg]:segment_boundary[seg + 1]])))

        results_vowel.write(',' + str(len(peaks)))

        text_file_1 = open(audiofile[:-4] + 'CH.csv', 'w')   # Opening CSV file to store results and to create TextGrid

        mark = []

        for seg in peaks:   # Extracting a 50 ms slice of the audio file based on the frame number
            mark.append(seg * hop_size)
            mark.append(seg * hop_size + window_size)

        for i in range(0, len(mark), 2):  # Writing the result to a CSV File
            text_file_1.write('%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + "Vowel" + "\n")

    except:
        results_vowel.write(audiofile + ',' + 'Error')


def fa_count(textfile):
    text_grid_1 = open(textfile, 'r')  # Open TextGrid in read mode
    data_1 = text_grid_1.read()  # Read the data from the TextGrid
    try:
        def count(vowel):  # Function to count the number of times the argument 'vowel' occurs in the TextGrid
            return data_1.count(vowel)

        c1 = count('"aa"')  # For vowel sound "aa"
        c2 = count('"AA"')  # For vowel sound "AA"
        c3 = count('"ae"')  # For vowel sound "ae"
        c4 = count('"aw"')  # For vowel sound "aw"
        c5 = count('"ay"')  # For vowel sound "ay"
        c6 = count('"ee"')  # For vowel sound "ee"
        c7 = count('"ex"')  # For vowel sound "ex"
        c8 = count('"ii"')  # For vowel sound "ii"
        c9 = count('"II"')  # For vowel sound "II"
        c10 = count('"oo"')  # For vowel sound "oo"
        c11 = count('"OO"')  # For vowel sound "OO"
        c12 = count('"oy"')  # For vowel sound "oy"
        c13 = count('"uu"')  # For vowel sound "uu"
        c14 = count('"UU"')  # For vowel sound "UU"

        c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14  # Summing up all the vowel occurrences

        start = data_1.find('xmin')  # The first occurrence of 'xmin', points to the staring time of the Audio File
        stop = data_1.find('xmax')  # The first occurrence of 'xmax', points to the ending time of the Audio File
        starting = float(data_1[start + 7])  # Extracting the starting time.(Always zero)
        stopping = float(
            data_1[stop + 7] + data_1[stop + 8] + data_1[stop + 9] + data_1[stop + 10] + data_1[stop + 11] + data_1[
                stop + 12])  # Extracting the stopping time.
        duration = stopping - starting  # Calculating the duration of the Audio File
        results_vowel.write(',' + str(c) + ',' + str(duration) + '\n')  # Writing the results in the csv file.

    except:
        results_vowel.write(',' + 'Error' + ',' + 'Error' + '\n')  # In case of error of execution of main code.


def textgrid(csvi):
    try:
        csvFileName = csvi
        TGFileName = csvFileName.split('.')[0] + '_NEW.TextGrid'  # Setting name of TextGrid file

        fid_csv = open(csvFileName, 'r')
        fidTG = open(TGFileName, 'w')

        reader = csv.reader(fid_csv, delimiter="\t")  # Reading data from csv file
        data = list(reader)  # Converting read data into python list format
        label_count = len(data)  # Finding total number of rows in csv file
        end_time = data[-1][1]

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

        for TG in range(label_count):
            fidTG.write('\t\tintervals [' + str(TG) + ']:\n')
            fidTG.write('\t\t\txmin = ' + str(data[TG][0]) + '\n')
            fidTG.write('\t\t\txmax = ' + str(data[TG][1]) + '\n')
            fidTG.write('\t\t\ttext = "' + str(data[TG][2]) + '"\n')

        fid_csv.close()
        fidTG.close()
    except:
        print csvi


def evaluation(textgrid1, textgrid2):
    try:
        text_grid_1 = open(textgrid1, 'r')  # Open the FA TextGrid
        text_grid_2 = open(textgrid2, 'r')  # Open the TextGrid created by the script

        data_1 = text_grid_1.read()  # Read and assign the content of the FA TextGrid to data_1
        data_2 = text_grid_2.read()  # Read and assign the content of the created TextGrid to data_2

        time_1 = []  # Creating an empty list to record time
        time_2 = []

        counter = 0
        for m in re.finditer('text = "', data_1):
            if data_1[m.start() - 33] == '=':
                time_1.append(float(
                    data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] + data_1[m.start() - 29] +
                    data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
                time_1.append(float(
                    data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[m.start() - 10] +
                    data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] + data_1[m.start() - 5]))
            else:
                time_1.append(float(
                    data_1[m.start() - 33] + data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] +
                    data_1[m.start() - 29] + data_1[m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
                time_1.append(float(
                    data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[m.start() - 10] +
                    data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[m.start() - 7] + data_1[m.start() - 6] + data_1[m.start() - 5]))

            if data_1[m.start() + 9] == '"':
                time_1.append(data_1[m.start() + 8])
            elif data_1[m.start() + 10] == '"':
                time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9])
            else:
                time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9] + data_1[m.start() + 10])

            time_1.append(counter)
            counter += 1


        for m in re.finditer('"Vowel"', data_2):
            time_2.append(float(
                data_2[m.start() - 34] + data_2[m.start() - 33] + data_2[m.start() - 32] + data_2[m.start() - 31] +
                data_2[m.start() - 30] + data_2[m.start() - 29]))
            time_2.append(float(
                data_2[m.start() - 17] + data_2[m.start() - 16] + data_2[m.start() - 15] + data_2[m.start() - 14] +
                data_2[m.start() - 13] + data_2[m.start() - 12]))

        listing = []

        for outer in range(0, len(time_2), 2):
            for inner in range(0, len(time_1), 4):
                if time_1[inner] <= time_2[outer] < time_1[inner + 1] and time_1[inner] < time_2[outer + 1] <= time_1[inner + 1]:
                    listing.append(time_1[inner + 2])
                    listing.append(time_1[inner + 3])

        for outer in range(0, len(time_2), 2):
            for inner in range(0, len(time_1) - 4, 4):
                if time_1[inner] < time_2[outer] < time_1[inner + 1] and time_1[inner + 3] < time_2[outer + 1] < time_1[inner + 4]:
                    listing.append(time_1[inner + 3])
                    listing.append(time_1[inner + 4])
                    listing.append(time_1[inner + 6])
                    listing.append(time_1[inner + 7])

        count = 0
        vowel_data = ['aa', 'AA', 'ae', 'aw', 'ay', 'ee', 'ex', 'ii', 'II', 'oo', 'OO', 'oy', 'uu', 'UU']

        already_here = []
        for vowel_sound in range(0, len(listing), 2):
            if listing[vowel_sound] in vowel_data and listing[vowel_sound + 1] not in already_here:
                count += 1
                already_here.append(listing[vowel_sound + 1])

        results_vowel.write(',' + str(count))

    except:
        results_vowel.write(',' + 'Error')

only_audio = glob.glob('F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Data 1\*.wav')  # Extract file name of all audio samples
only_text = glob.glob('F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Data 1\*.TextGrid')  # Extract file name of all force aligned TextGrid

for j in only_audio:
    peak_elimination(j)  # Run Peak elimination algorithm
    textgrid(str(j[:-4] + 'PE.csv'))  # Create TextGrid based on results of peak elimination
    evaluation(str(j[:-4] + '.TextGrid'), str(j[:-4] + 'PE_NEW.TextGrid'))  # Evaluate the TextGrid created and the force aligned TextGrid
    convexhull(j)  # Run the convex hull algorithm
    textgrid(str(j[:-4] + 'CH.csv'))  # Create TextGrid based on results of convex hull
    evaluation(str(j[:-4] + '.TextGrid'), str(j[:-4] + 'CH_NEW.TextGrid'))  # Evaluate the TextGrid created and force aligned TextGrid
    fa_count(str(j[:-4] + '.TextGrid'))  # Count the number of vowels in the audio file according to the force aligned TextGrid


print datetime.now() - startTime  # Print program run time
winsound.Beep(300, 2000)
