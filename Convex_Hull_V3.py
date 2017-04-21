from scipy.io import wavfile
import math
import matplotlib.pyplot as plt
import numpy as np


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
    # precomputed coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

audiofile = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation CH V2\Analyze\Vowel_Evaluation_V2_I1\\6.wav'


def convex_hull_analysis(audio_file, window_dur, hop_dur, threshold_smooth, threshold):
    fs, data = wavfile.read(audio_file)  # Reading data from wav file in an array
    data = data / float(2 ** 15)  # Normalizing it to [-1,1] range from [-2^15,2^15]
    window_size = int(window_dur * fs * 0.001)  # Converting window length to samples
    hop_size = int(hop_dur * fs * 0.001)  # Converting hop length to samples
    window_type = np.hanning(window_size)  # Window type: Hanning (by default)
    no_frames = int(math.ceil(len(data) / (float(hop_size))))  # Determining the number of frames
    zero_array = np.zeros(window_size)  # Appending appropriate number of zeros
    data = np.concatenate((data, zero_array))

    st_energy = [0] * no_frames
    auto = []
    # Calculating frame wise short term energy
    for i in range(no_frames):
        frame = data[i * hop_size:i * hop_size + window_size] * window_type
        st_energy[i] = sum(frame ** 2)
        auto.append(max(np.convolve(frame, frame)))

    auto = moving_average(auto,20)
    seg = int(no_frames / 200)
    rem = no_frames % 200
    print seg, rem, no_frames

    for i in range(0, seg - 1):

        maxi = max(auto[i * 200:(i + 1) * 200])
        for j in range(i * 200, (i + 1) * 200):
            auto[j] = auto[j] / maxi

    auto = auto.tolist()
    convex_hull = []
    segment_boundary = [0]


    def segmentation(frame_energy):
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

        # plt.plot(frame_energy, 'red', label='Short term energy')
        # plt.plot(convex_hull, 'black', label='Convex Hull')
        # plt.xlabel('Frame Number')
        # plt.ylabel('Magnitude')
        # plt.title('Short term energy and Convex Hull')
        # plt.legend()
        # plt.show()

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

        # plt.plot(frame_energy, 'red', label='Short term energy')
        # plt.plot(convex_hull, 'black', label='Modified Convex Hull')
        # plt.xlabel('Frame Number')
        # plt.ylabel('Magnitude')
        # plt.title('Short term energy and Modified Convex Hull')
        # plt.legend()
        # plt.show()

        frame_energy.reverse()  # Reverse frame_energy once more for correct indexing in proceeding steps
        diff_b.reverse()  # Reverse diff_b for correct indexing in proceeding steps

        if max(diff_f) >= max(diff_b):  # Comparing Maximum's
            if max(diff_f) > threshold:  # Comparing the diff with a unified threshold which decides whether the difference is large enough or not to warrant a segment
                st = auto.index(frame_energy[0])  # Starting index of segment 1
                bp = auto.index((frame_energy[diff_f.index(max(diff_f))]))  # Ending index of segment 1, and starting index of segment 2
                sp = auto.index(frame_energy[-1])  # Ending index of segment 2
                segment_boundary.append(bp)  # Adding the breakpoint to the segment boundary list
                if len(auto[st:bp]) > 0 and len(auto[bp:sp]) > 0:  # If both segments are larger than 0 in length, then proceed with further segmentation
                    segmentation(auto[st:bp])
                    segmentation(auto[bp:sp])
                else:
                    return ()
            else:
                return ()
        else:
            if max(diff_b) > threshold:
                st = auto.index(frame_energy[0])
                bp = auto.index((frame_energy[diff_b.index(max(diff_b))]))
                sp = auto.index(frame_energy[-1])
                segment_boundary.append(bp)
                if len(auto[st:bp]) > 0 and len(auto[bp:sp]) > 0:
                    segmentation(auto[st:bp])
                    segmentation(auto[bp:sp])
                else:
                    return ()
            else:
                return ()


    ########################################################################################################################

    ########################################################################################################################
    segmentation(auto)

    #######################################################################################################################
    # What : Since the boundaries are not determined in a sequential order, they need to be sorted before we can find
    # the peaks in each segment

    segment_boundary.append(len(auto))
    segment_boundary.sort()

    #######################################################################################################################
    # What : In each of the segments determined by the convex hull algorithm, we find the peak and mark that as the vowel centre.
    peaks = []
    for j in range(0, len(segment_boundary) - 1):
        if len(auto[segment_boundary[j]:segment_boundary[j + 1]]) != 0:
            peaks.append(auto.index(max(auto[segment_boundary[j]:segment_boundary[j + 1]])))

    print "Vowel Approx : ", len(peaks)

    #######################################################################################################################
    # What: CSV file creation for Making Text_Grid
    text_file_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\R3_Mark.csv', 'w')

    mark = []
    for j in peaks:
        mark.append(j * hop_size)
        mark.append(j * hop_size + window_size)

    for i in range(0, len(mark), 2):
        text_file_1.write(
            '%06.3f' % (mark[i] * 0.0000625) + "\t" + '%06.3f' % (mark[i + 1] * 0.0000625) + "\t" + "Vowel" + "\n")


    #######################################################################################################################
    # What : Plotting the peaks and segments
    plt.plot(auto, 'red', label='Periodicity')
    # plt.plot(st_energy)
    # plt.plot(st_energy_1, 'blue')
    # plt.vlines(peaks[0], min(st_energy), max(st_energy), 'green', label='Peak', linestyles='dashed')
    # for j in peaks:
    #     plt.vlines(j, min(st_energy), max(st_energy), 'green', linestyles='dashed')
    plt.vlines(segment_boundary[0], min(auto), max(auto), 'black', label='Segment boundary')
    for j in segment_boundary:
        plt.vlines(j, min(auto), max(auto), 'black')
    plt.ylabel('Magnitude')
    plt.xlabel('Frame Number')
    plt.title('Segments')
    plt.legend()
    plt.show()

convex_hull_analysis(audiofile, 30, 5, 200, 0.1)
