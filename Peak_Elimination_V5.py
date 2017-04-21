"""
This script can be used to find the vowel candidates in an audio file.
It uses a peak elimination technique based on threshold values and by determining
the strength of peaks.

Output : A csv file storing the starting and ending time stamps of the vowel sound. Use the TextGrid script to
generate a text grid file which can be viewed in Praat.

Author: Rishabh Brajabasi
Last updated: 02nd February 2017
"""

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
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# audiofile = 'F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Newmix\\00001laxmi_56580b937e63f5035c003431_57b19ede9ee20a03d87b5f89_2_002000200000111.wav'
# audiofile = 'C:\Users\Mahe\Desktop\\17.wav'
audiofile ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V3\Analyze\Vowel_Evaluation_V3_I6_M12\\17.wav'
def peak_elimination_analysis(audio_file, window_dur, hop_dur, degree, ripple_upper_limit, ripple_lower_limit, threshold_smooth):

    fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
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
    plt.xlim(0,x_values[-1])
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude')
    plt.title('Sound Waveform')
    plt.show()
    #######################################################################################################################
    noise_energy = 0
    energy = [0] * length

    # Calculating Noise energy
    # What : Taking the first 800 samples of the original sound file and averaging it. Using this as
    #       average noise energy for threshold application.
    # Why : So that peaks caused by noise are better eliminated.
    for j in range(length):
        energy[j] = data[j] * data[j]
    for j in range(0, 800):  # energy
        noise_energy += energy[j]
    noise_energy /= 800
    #######################################################################################################################

    #######################################################################################################################
    st_energy = [0] * no_frames

    # Calculating frame wise short term energy
    for i in range(no_frames):
        frame = data[i * hop_size:i * hop_size + window_size] * window_type
        st_energy[i] = sum(frame ** 2)

    if len(st_energy) < threshold_smooth:
        original_st_energy = st_energy
        st_energy = st_energy
        st_energy_ma = st_energy
    else:
        original_st_energy = st_energy
        st_energy = savitzky_golay(st_energy, 51, degree)  # window size 51, polynomial order 3
        st_energy_ma = movingaverage(st_energy, 25)
    # Why : Higher the degree of the polynomial, more accurate recreation of the original curve.
    #######################################################################################################################
    plt.figure('Original Short Term Energy')
    plt.plot(original_st_energy)
    plt.xlim(0,len(original_st_energy))
    plt.ylim(min(original_st_energy),max(original_st_energy))
    plt.ylabel('Magnitude')
    plt.xlabel('Frame Number')
    plt.title('Short Term Energy')
    plt.show()

    plt.figure('Short Term Energy')
    plt.plot(st_energy, 'red')
    plt.xlim(0,len(st_energy))
    plt.ylim(min(st_energy),max(st_energy))
    plt.ylabel('Magnitude')
    plt.xlabel('Frame Number')
    plt.title('Smoothed Short Term Energy')
    plt.show()

    plt.figure('Short Term Energy')
    plt.plot(st_energy, 'red', label='Smoothed')
    plt.plot(original_st_energy, 'blue', label='Original')
    # plt.plot(st_energy_ma, 'black')
    plt.xlim(0,len(st_energy))
    plt.ylabel('Magnitude')
    plt.xlabel('Frame Number')
    plt.title('Comparison of original and smoothed')
    plt.legend()
    plt.show()

    plt.figure('STE')
    plt.subplot(211)
    plt.plot(x_values,data)
    plt.xlim(0,x_values[-1])
    plt.title('Sound Waveform')
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude')
    plt.subplot(212)
    plt.plot(original_st_energy)
    plt.xlim(0,len(original_st_energy))
    plt.title('Original Short Term Energy')
    plt.xlabel('Frame No.')
    plt.ylabel('Magnitude')
    # plt.subplot(313)
    # plt.plot(st_energy,'red')
    # plt.xlim(0,len(st_energy))
    # plt.title('Smoothed Short Term energy')
    plt.show()
    #######################################################################################################################
    # Finding the peaks of the short term energy
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
    print "Number of peaks                        :", count_of_peaks

    #######################################################################################################################
    plt.figure('Peaks')
    plt.plot(st_energy, 'red', label='Short term energy')
    plt.plot(peak, 'green', label='Peaks')
    plt.xlabel('Frame number')
    plt.ylabel('Magnitude')
    plt.title('Peaks')
    plt.legend()
    plt.xlim(0,len(st_energy))
    plt.ylim(min(st_energy),max(st_energy))
    plt.text(500, 3.5, 'No of peaks : ' + str(count_of_peaks), bbox={'facecolor':'red', 'alpha':1, 'pad':10})
    plt.show()


    #######################################################################################################################
    # What : The threshold which eliminates minor peaks.
    threshold = 0.01 + 0.04 * (noise_energy + (sum(peak) / count_of_peaks))
    #######################################################################################################################

    #######################################################################################################################
    # What: The peaks after the one's which are less than the threshold are removed
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
    #######################################################################################################################
    thresh = []
    for j in range(len(peak_threshold)):
        thresh.append(threshold)

    plt.figure('Peaks after threshold')
    plt.plot(st_energy, 'red', label='Short term energy')
    plt.plot(thresh, 'black', label='Threshold')
    plt.plot(peak_threshold, 'green', label='Peaks')
    plt.xlabel('Frame number')
    plt.ylabel('Magnitude')
    plt.title('Peaks after threshold')
    plt.xlim(0,len(st_energy))
    plt.ylim(min(st_energy), max(st_energy))
    plt.legend()
    plt.text(500, 3.5, 'No of peaks : ' + str(count_of_peaks_threshold), bbox={'facecolor':'red', 'alpha':1, 'pad':10})
    plt.show()


    #######################################################################################################################
    # What : Finding the valleys on the short term energy curve
    valley = []
    count_of_valleys = 0
    location_valley = []

    for p in range(len(st_energy)):
        if p == 0:
            if st_energy[p] < st_energy[p + 1]:
                valley.append(st_energy[p])
                count_of_valleys += 1
                location_valley.append(p)
            else:
                valley.append(0)
        elif p == len(st_energy) - 1:
            if st_energy[p] < st_energy[p - 1]:
                valley.append(st_energy[p])
                count_of_valleys += 1
                location_valley.append(p)
            else:
                valley.append(0)
        else:
            if st_energy[p] < st_energy[p + 1] and st_energy[p] < st_energy[p - 1]:
                valley.append(st_energy[p])
                count_of_valleys += 1
                location_valley.append(p)
            else:
                valley.append(0)
    #######################################################################################################################
    #######################################################################################################################


    # What : Calculating the significant peaks and valleys needed to calculate the relative strength of each peak, as a strong,
    # or a weak peak. For this we calculate the ripple value of each peak.
    location = location_peak + location_valley
    location.sort()
    ripple_valley = []
    ripple_peak = []
    ripple = []

    for k in range(len(location_peak)):
        q = location.index(location_peak[k])
        if location_peak[k] == len(peak) - 1:
            ripple.append(location[q - 1])
            ripple_valley.append(location[q - 1])
            ripple.append(location[q])
            ripple_peak.append(location[q])
            ripple.append(location[q - 1])
            ripple_valley.append(location[q - 1])
        elif location_peak[k] == 0:
            ripple.append(location[q + 1])
            ripple_valley.append(location[q + 1])
            ripple.append(location[q])
            ripple_peak.append(location[q])
            ripple.append(location[q + 1])
            ripple_valley.append(location[q + 1])
        else:
            ripple.append(location[q - 1])
            ripple_valley.append(location[q - 1])
            ripple.append(location[q])
            ripple_peak.append(location[q])
            ripple.append(location[q + 1])
            ripple_valley.append(location[q + 1])

    value_valley = []
    for j in range(len(ripple_valley)):
        value_valley.append(st_energy[ripple_valley[j]])

    ripple_value = []
    for k in range(1, len(ripple), 3):
        ripple_value.append((st_energy[ripple[k]] - st_energy[ripple[k + 1]]) / (st_energy[ripple[k]] - st_energy[ripple[k - 1]]))
    #######################################################################################################################



    #######################################################################################################################
    # What : Shortlist peaks based on their ripple value.
    ripple_value_thresh = []
    count_of_vowels = 0
    # print location_peak
    # print location_valley
    # print location
    print ripple_peak
    print ripple_valley
    print ripple
    print ripple_value


    for k in range(len(ripple_value)):
        print ripple_peak[ripple_value.index(ripple_value[k])]
        if k != len(ripple_value)-1:
            if ripple_upper_limit >= ripple_value[k] >= ripple_lower_limit:
                ripple_value_thresh.append(ripple_value[k])
                count_of_vowels += 1
            if ripple_peak[ripple_value.index(ripple_value[k+1])] - ripple_peak[ripple_value.index(ripple_value[k])] < 20:
                if ripple_value[k] >= ripple_upper_limit and ripple_value[k+1] <= ripple_lower_limit:
                    if ripple_value[k+1] < 0.1:
                        ripple_value_thresh.append(ripple_value[k])
                        count_of_vowels += 1
                    else:
                        ripple_value_thresh.append(ripple_value[k])
                        ripple_value_thresh.append(ripple_value[k+1])
                        count_of_vowels += 2
                elif ripple_value[k] <= ripple_lower_limit and ripple_value[k+1] >= ripple_upper_limit:
                    if ripple_value[k] < 0.1:
                        ripple_value_thresh.append(ripple_value[k])
                        count_of_vowels += 1
                    else:
                        ripple_value_thresh.append(ripple_value[k])
                        ripple_value_thresh.append(ripple_value[k+1])
                        count_of_vowels += 2
                else:
                    ripple_value_thresh.append(0)
        else:
            if ripple_upper_limit >= ripple_value[k] >= ripple_lower_limit:
                ripple_value_thresh.append(ripple_value[k])
                count_of_vowels += 1
            else:
                ripple_value_thresh.append(0)


    print "Vowel Approx                           :", count_of_vowels
    #######################################################################################################################
    # What : Evaluating final location of selected peaks, so that we can create a Text_Grid

    loc = []
    peak_threshold[:] = []
    for j in range(len(ripple_value_thresh)):
        if ripple_value_thresh[j] != 0:
            loc.append(location_peak[ripple_value.index(ripple_value_thresh[j])])

    print loc

    for j in range(no_frames):
        if j in loc:
            peak_threshold.append(st_energy[loc.index(j)])
        else:
            peak_threshold.append(0)
    #######################################################################################################################

    #######################################################################################################################
    # What: CSV file creation for Making Text_Grid
    text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\R3_Mark.csv', 'w')

    mark = []
    for j in range(len(peak_threshold)):
        if peak_threshold[j] is not 0:
            mark.append(j * hop_size)
            mark.append(j * hop_size + window_size)

    for i in range(0, len(mark), 2):
        text_file_1.write(
            '%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + "Vowel" + "\n")
    #######################################################################################################################

    #######################################################################################################################
    # What : Plotting the various Curves
    #
    plt.figure('Short Term Energy')
    plt.plot(st_energy, 'black', label='Short term energy')
    plt.scatter(location_peak, value_peak, color='red', label='Peak')
    plt.scatter(ripple_valley, value_valley, color='green', label='Valley')
    for j in range(len(location_peak)):
        plt.text(location_peak[j], value_peak[j], str(round(ripple_value[j], 2)))
    plt.xlabel('Frame number')
    plt.ylabel('Magnitude')
    plt.title('Ripple Value')
    plt.xlim(0,len(st_energy))
    plt.ylim(min(st_energy),max(st_energy))
    plt.legend()
    plt.show()

    plt.figure('Peaks after threshold')
    plt.plot(st_energy, 'red', label='Short term energy')
    for j in loc:
        plt.vlines(j, 0, max(st_energy), 'black')
    plt.xlabel('Frame number')
    plt.ylabel('Magnitude')
    plt.title('Peaks after ripple calculation')
    plt.legend()
    plt.xlim(0, len(st_energy))
    plt.ylim(min(st_energy), max(st_energy))
    plt.text(500, 3.5, 'No of peaks : ' + str(count_of_vowels), bbox={'facecolor':'red', 'alpha': 1, 'pad': 10})
    plt.show()
    print len(st_energy)
    ########################################################################################################################

peak_elimination_analysis(audiofile, 50, 7, 3, 2.0, 0.4, 120)
