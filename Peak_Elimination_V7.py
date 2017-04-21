from scipy.io import wavfile
import math
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
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
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

# audiofile = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Data 1\\00001laxmi_56580b937e63f5035c003431_57b19ede9ee20a03d87b5f8b_16_04011000.wav'
# audiofile = 'C:\Users\Mahe\Desktop\\test2.wav'
audiofile ='F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V4\Analyze\Vowel_Evaluation_V4_I2\\98.wav'
#
#
def peak_elimination_analysis(audio_file, window_dur, hop_dur, degree, ripple_upper_limit, ripple_lower_limit, threshold_smooth):
    fs, data = wavfile.read(audio_file)
    data = data / float(2 ** 15)
    window_size = int(window_dur * fs * 0.001)
    hop_size = int(hop_dur * fs * 0.001)
    window_type = np.hanning(window_size)
    no_frames = int(math.ceil(len(data) / (float(hop_size))))
    zero_array = np.zeros(window_size)
    data = np.concatenate((data, zero_array))
    length = len(data)
    x_values = np.arange(0, len(data), 1) / float(fs)

#----------------------------------------------------------------------------------------------------------------------#
    plt.figure('Sound Waveform')
    plt.plot(x_values, data)
    plt.xlabel('Time in seconds')
    plt.ylabel('Amplitude')
    plt.title('Sound Waveform')
    plt.show()
    #----------------------------------------------------------------------------------------------------------------------#
    noise_energy = 0
    energy = [0] * length
    for j in range(length):
        energy[j] = data[j] * data[j]
    for j in range(0, 800):  # energy
        noise_energy += energy[j]
    noise_energy /= 800
    # ----------------------------------------------------------------------------------------------------------------------#
    st_energy = []
    for i in range(no_frames):
        frame = data[i * hop_size:i * hop_size + window_size] * window_type
        st_energy.append(sum(frame ** 2))

    norm_max_square = max(st_energy)
    for i in range(no_frames):
        st_energy[i] = st_energy[i] / norm_max_square
    # ----------------------------------------------------------------------------------------------------------------------#
    o_peak = []
    o_loc = []
    for p in range(len(st_energy)):
        if p == 0:
            if st_energy[p] > st_energy[p + 1]:
                o_peak.append(st_energy[p])
                o_loc.append(p)
            else:
                o_peak.append(0)
        elif p == len(st_energy) - 1:
            if st_energy[p] > st_energy[p - 1]:
                o_peak.append(st_energy[p])
                o_loc.append(p)
            else:
                o_peak.append(0)
        else:
            if st_energy[p] > st_energy[p + 1] and st_energy[p] > st_energy[p - 1]:
                o_peak.append(st_energy[p])
                o_loc.append(p)
            else:
                o_peak.append(0)
    # ---------------------------------------------------------------------------------------------------------------------#
    if len(st_energy) < threshold_smooth:
        original_st_energy = st_energy
        st_energy = st_energy
    else:
        original_st_energy = st_energy
        st_energy = movingaverage(st_energy,5)
        # st_energy = savitzky_golay(st_energy, 51, degree)  # window size 51, polynomial order 3
    # ----------------------------------------------------------------------------------------------------------------------#
    plt.figure('Original Short Term Energy')
    plt.plot(original_st_energy)
    plt.ylabel('Magnitude')
    plt.xlabel('Frame Number')
    plt.title('Short Term Energy')
    plt.show()

    plt.figure('Short Term Energy')
    plt.plot(st_energy, 'red')
    plt.ylabel('Magnitude')
    plt.xlabel('Frame Number')
    plt.title('Smoothed Short Term Energy')
    plt.show()

    plt.figure('Short Term Energy')
    plt.plot(st_energy, 'red', label='Smoothed')
    plt.plot(original_st_energy, 'blue', label='Original')
    plt.ylabel('Magnitude')
    plt.xlabel('Frame Number')
    plt.title('Comparison of original and smoothed')
    plt.legend()
    plt.show()
    # ----------------------------------------------------------------------------------------------------------------------#
    peak = []
    t_peak = []
    count_of_peaks = 0
    for p in range(len(st_energy)):
        if p == 0:  # First element
            if st_energy[p] > st_energy[
                        p + 1]:  # If the first element is greater than the succeeding element it is a peak.
                peak.append(st_energy[p])  # Append the energy level of the peak
                t_peak.append(st_energy[p])  # Append the energy level of the peak
                count_of_peaks += 1  # Increment count
            else:
                peak.append(0)  # Else append a zero
        elif p == len(st_energy) - 1:  # Last element
            if st_energy[p] > st_energy[
                        p - 1]:  # If the last element is greater than the preceding element it is a peak.
                peak.append(st_energy[p])  # Append the energy level of the peak
                t_peak.append(st_energy[p])  # Append the energy level of the peak
                count_of_peaks += 1  # Increment count
            else:
                peak.append(0)  # Else append a zero
        else:  # All the other elements
            if st_energy[p] > st_energy[p + 1] and st_energy[p] > st_energy[
                        p - 1]:  # If the element is greater than the element preceding and succeeding it, it is a peak.
                peak.append(st_energy[p])  # Append the energy level of the peak
                t_peak.append(st_energy[p])  # Append the energy level of the peak
                count_of_peaks += 1  # Increment count
            else:
                peak.append(0)  # Else append a zero
    print "Number of peaks                        :", count_of_peaks

    plt.figure('Peaks')
    plt.plot(st_energy, 'red', label='Short term energy')
    plt.plot(peak, 'green', label='Peaks')
    plt.xlabel('Frame number')
    plt.ylabel('Magnitude')
    plt.title('Peaks')
    plt.legend()
    plt.text(1400, 10, 'No of peaks : ' + str(count_of_peaks), bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()


    threshold = (sum(t_peak)/len(t_peak))

    # threshold = 0.01 + 0.04 * (noise_energy + (sum(peak) / count_of_peaks))
    print threshold
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
    print peak
    print "Peaks after applying threshold         :", count_of_peaks_threshold

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
    plt.legend()
    plt.text(1400, 10, 'No of peaks : ' + str(count_of_peaks_threshold),
             bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()

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
        ripple_value.append(
            (st_energy[ripple[k]] - st_energy[ripple[k + 1]]) / (st_energy[ripple[k]] - st_energy[ripple[k - 1]]))

    ripple_value_cond_2 = []
    for k in range(1, len(ripple), 3):
        ripple_value_cond_2.append(
            (st_energy[ripple[k]] - st_energy[ripple[k - 1]] / st_energy[ripple[k]]))

    ripple_value_cond_3 = []
    for k in range(1, len(ripple), 3):
        ripple_value_cond_3.append(
            (st_energy[ripple[k]] - st_energy[ripple[k + 1]] / st_energy[ripple[k]]))

    # ripple_value_thresh = []
    count_of_vowels = 0
    print 'LP', location_peak
    # print location_valley
    # print location
    print 'RP', ripple_peak
    # print ripple_valley
    # print ripple
    # print ripple_value
    loc = []

    for k in range(len(ripple_value)):
        loc.append(location_peak[ripple_value.index(ripple_value[k])])

    print 'PEAKS 1:', loc

    for k in range(len(ripple_value)):
        if k != len(ripple_value)-1:
            if ripple_value[k] > 3.0 and ripple_value[k + 1] < 1.4 or ripple_value[k] > 1.02 and ripple_value[k + 1] < \
                    0.3:
                v1 = st_energy[location_peak[ripple_value.index(ripple_value[k])]]
                v2 = st_energy[location_peak[ripple_value.index(ripple_value[k + 1])]]
                if v1 >= v2:
                    loc.remove(location_peak[ripple_value.index(ripple_value[k+1])])
                else:
                    loc.remove(location_peak[ripple_value.index(ripple_value[k])])
        else:
            if ripple_value[k] > 3.0:
                loc.remove(location_peak[ripple_value.index(ripple_value[k])])

    print 'PEAKS 2:', loc

    # for k in range(len(ripple_value_cond_2)):
    #     if ripple_value_cond_2[k] < 0.3 and ripple_value_cond_3[k] < 0.3:
    #         if location_peak[ripple_value_cond_2.index(ripple_value_cond_2[k])] in loc:
    #             loc.remove(location_peak[ripple_value_cond_2.index(ripple_value_cond_2[k])])

    print 'PEAKS F:', loc
    print 'No. of vowels:', len(loc)
    print 'Original:', o_loc
    loc_1 = []
    set = 0
    for val in loc:
        for i in range(val-10, val+10, 1):
            if i in o_loc and set == 0:
                loc_1.append(i)
                set = 1
        set = 0


    for j in range(no_frames):
        if j in loc_1:
            peak_threshold.append(original_st_energy[loc_1.index(j)])
        else:
            peak_threshold.append(0)
    print 'PEAKS O:', loc_1
    plt.plot(original_st_energy)
    for j in loc_1:
        plt.vlines(j, 0, max(original_st_energy), 'black')
    plt.show()
    text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\R3_Mark.csv', 'w')

    mark = []
    for j in range(len(peak_threshold)):
        if peak_threshold[j] is not 0:
            mark.append(j * hop_size)
            mark.append(j * hop_size + window_size)

    for i in range(0, len(mark), 2):
        text_file_1.write(
            '%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + "Vowel" + "\n")

    plt.figure('Short Term Energy')
    plt.plot(st_energy, 'black', label='Short term energy')
    plt.scatter(location_peak, value_peak, color='red', label='Peak')
    plt.scatter(ripple_valley, value_valley, color='green', label='Valley')
    for j in range(len(location_peak)):
        plt.text(location_peak[j], value_peak[j], str(round(ripple_value[j], 2)))
    plt.xlabel('Frame number')
    plt.ylabel('Magnitude')
    plt.title('Ripple Value')
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
    plt.text(1400, 10, 'No of peaks : ' + str(count_of_vowels), bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()

peak_elimination_analysis(audiofile, 50, 7, 10, 2.0, 0.4, 200)
