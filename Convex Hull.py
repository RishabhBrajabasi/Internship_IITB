from scipy.io import wavfile
from operator import sub
import math
import matplotlib.pyplot as plt
import numpy as np


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # type: (object, object, object, object, object) -> object
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    """
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


########################################################################################################################
audio_file = 'F:\Projects\Active Projects\Project Intern_IITB\Convex Hull\Test_accuracy.wav'
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
length = len(data)  # Finding length of the actual data
########################################################################################################################

########################################################################################################################
st_energy = [0] * no_frames
maximum = [0] * no_frames
frame_number = [0] * no_frames
start = [0] * no_frames
stop = [0] * no_frames

# Calculating frame wise short term energy
for i in range(no_frames):
    frame = data[i * hop_size:i * hop_size + window_size] * window_type
    start[i] = i * hop_size
    stop[i] = i * hop_size + window_size
    st_energy[i] = sum(frame ** 2)

original_st_energy = st_energy
# st_energy = savitzky_golay(st_energy, 51, 10)  # window size 51, polynomial order 3
# st_energy = st_energy.tolist()

convex_hull = []
segment_boundary = []


def segmentation(frame_energy):
    convex_hull[:] = []


    break_point = frame_energy.index(max(frame_energy))
    last_1 = frame_energy[0]
    last_2 = frame_energy[break_point]

    for j in range(len(frame_energy) - 1):

        if j < break_point:
            if frame_energy[j] > frame_energy[j - 1] and frame_energy[j] > last_1:
                convex_hull.append(frame_energy[j])
                last_1 = frame_energy[j]
            else:
                convex_hull.append(last_1)

        elif j > break_point:
            if frame_energy[j + 1] < frame_energy[j] < last_2:
                convex_hull.append(frame_energy[j])
                last_2 = frame_energy[j + 1]
            else:
                convex_hull.append(last_2)

        else:
            convex_hull.append(frame_energy[break_point])

    convex_hull.append(last_2)

    plt.plot(frame_energy, 'green')
    plt.plot(convex_hull, 'red')



    diff = list(map(sub, convex_hull, frame_energy))
    for j in range(len(diff)):
        if diff[j] < 0:
            diff[j] *= -0

    st = st_energy.index(frame_energy[0])
    bp = st_energy.index(frame_energy[diff.index(max(diff))])
    sp = st_energy.index(frame_energy[-1])

    plt.scatter(bp, max(frame_energy))
    plt.scatter(bp, 0)
    plt.scatter(st, max(frame_energy))
    plt.scatter(sp, max(frame_energy))
    plt.show()

    segment_boundary.append(bp)

    if convex_hull == frame_energy:
        return()

    if max(diff) > 0.2 and bp != sp and bp != st:
        segmentation(st_energy[st:bp])
        segmentation(st_energy[bp:sp])

    else:
        return ()


segmentation(st_energy)

segments = [max(st_energy)]
for k in range(1, len(st_energy)-1):
    if k in segment_boundary:
        segments.append(max(st_energy))
    else:
        segments.append(0)
segments.append(max(st_energy))

print segment_boundary
plt.figure("Short Term Energy")
plt.plot(st_energy)
plt.stem(segments, 'black')


# plt.figure("First Run")
# plt.subplot(211)
# plt.plot(seg_1)
# plt.subplot(212)
# plt.plot(seg_2)
#
# plt.figure("First Half")
# plt.subplot(211)
# plt.plot(seg_1_1)
# plt.subplot(212)
# plt.plot(seg_1_2)
#
# plt.figure("Second Half")
# plt.subplot(211)
# plt.plot(seg_2_1)
# plt.subplot(212)
# plt.plot(seg_2_2)
#
plt.show()
