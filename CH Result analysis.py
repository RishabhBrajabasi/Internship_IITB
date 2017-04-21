"""

"""

import matplotlib.pyplot as plt

data = open('F:\Projects\Active Projects\Project Intern_IITB\Parameter Optimization CH V3\Vowel_opt.csv', 'r')
values = data.read()
lists = values.split('\n')
lists.pop(0)

work = []
for i in range(len(lists)):
    work.append(lists[i].split(','))

val = []


def plot_threshold_precision(start=0, window_duration='20ms'):
    x1 = [work[i][2] for i in range(start, start+5, 1)]
    y1 = [work[i][3] for i in range(start, start+5, 1)]
    x2 = [work[i][2] for i in range(start+5, start+10, 1)]
    y2 = [work[i][3] for i in range(start+5, start+10, 1)]
    x3 = [work[i][2] for i in range(start+10, start+15, 1)]
    y3 = [work[i][3] for i in range(start+10, start+15, 1)]
    x4 = [work[i][2] for i in range(start+15, start+20, 1)]
    y4 = [work[i][3] for i in range(start+15, start+20, 1)]

    plt.plot(x1, y1, color='red', label='Hop Duration 3ms')
    plt.scatter(x1, y1, color='red')
    plt.plot(x2, y2, color='blue', label='Hop Duration 5ms')
    plt.scatter(x2, y2, color='blue')
    plt.plot(x3, y3, color='green', label='Hop Duration 7ms')
    plt.scatter(x3, y3, color='green')
    plt.plot(x4, y4, color='black', label='Hop Duration 10ms')
    plt.scatter(x4, y4, color='black')

    plt.title(window_duration)
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def plot_threshold_recall(start=0, window_duration='20ms'):
    x_p_20_3 = [work[i][2] for i in range(start, start+5, 1)]
    y_p_20_3 = [work[i][4] for i in range(start, start+5, 1)]
    x_p_20_5 = [work[i][2] for i in range(start+5, start+10, 1)]
    y_p_20_5 = [work[i][4] for i in range(start+5, start+10, 1)]
    x_p_20_7 = [work[i][2] for i in range(start+10, start+15, 1)]
    y_p_20_7 = [work[i][4] for i in range(start+10, start+15, 1)]
    x_p_20_10 = [work[i][2] for i in range(start+15, start+20, 1)]
    y_p_20_10 = [work[i][4] for i in range(start+15, start+20, 1)]

    plt.plot(x_p_20_3, y_p_20_3, color='red', label='Hop Duration 3ms')
    plt.scatter(x_p_20_3, y_p_20_3,color='red')
    plt.plot(x_p_20_5, y_p_20_5, color='blue', label='Hop Duration 5ms')
    plt.scatter(x_p_20_5, y_p_20_5, color='blue')
    plt.plot(x_p_20_7, y_p_20_7, color='green', label='Hop Duration 7ms')
    plt.scatter(x_p_20_7, y_p_20_7, color='green')
    plt.plot(x_p_20_10, y_p_20_10, color='black', label='Hop Duration 10ms')
    plt.scatter(x_p_20_10, y_p_20_10, color='black')

    plt.title(window_duration)
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()


def plot_window_duration_precision(start=0, hop_duration='3ms'):
    x1 = [work[i][0] for i in range(start, start+80, 20)]
    y1 = [work[i][3] for i in range(start, start+80, 20)]
    x2 = [work[i][0] for i in range(start+1, start+80, 20)]
    y2 = [work[i][3] for i in range(start+1, start+80, 20)]
    x3 = [work[i][0] for i in range(start+2, start+80, 20)]
    y3 = [work[i][3] for i in range(start+2, start+80, 20)]
    x4 = [work[i][0] for i in range(start+3, start+80, 20)]
    y4 = [work[i][3] for i in range(start+3, start+80, 20)]
    x5 = [work[i][0] for i in range(start+4, start+80, 20)]
    y5 = [work[i][3] for i in range(start+4, start+80, 20)]

    plt.plot(x1, y1, color='red', label='Threshold 0.1')
    plt.scatter(x1, y1,color='red')
    plt.plot(x2, y2, color='blue', label='Threshold 0.2')
    plt.scatter(x2, y2, color='blue')
    plt.plot(x3, y3, color='green', label='Threshold 0.3')
    plt.scatter(x3, y3, color='green')
    plt.plot(x4, y4, color='black', label='Threshold 0.4')
    plt.scatter(x4, y4, color='black')
    plt.plot(x5, y5, color='yellow', label='Threshold 0.5')
    plt.scatter(x5, y5, color='yellow')

    plt.title(hop_duration)
    plt.xlabel('Window Duration')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def plot_window_duration_recall(start=0, hop_duration='3ms'):
    x1 = [work[i][0] for i in range(start, start+80, 20)]
    y1 = [work[i][4] for i in range(start, start+80, 20)]
    x2 = [work[i][0] for i in range(start+1, start+80, 20)]
    y2 = [work[i][4] for i in range(start+1, start+80, 20)]
    x3 = [work[i][0] for i in range(start+2, start+80, 20)]
    y3 = [work[i][4] for i in range(start+2, start+80, 20)]
    x4 = [work[i][0] for i in range(start+3, start+80, 20)]
    y4 = [work[i][4] for i in range(start+3, start+80, 20)]
    x5 = [work[i][0] for i in range(start+4, start+80, 20)]
    y5 = [work[i][4] for i in range(start+4, start+80, 20)]

    plt.plot(x1, y1, color='red', label='Threshold 0.1')
    plt.scatter(x1, y1, color='red')
    plt.plot(x2, y2, color='blue', label='Threshold 0.2')
    plt.scatter(x2, y2, color='blue')
    plt.plot(x3, y3, color='green', label='Threshold 0.3')
    plt.scatter(x3, y3, color='green')
    plt.plot(x4, y4, color='black', label='Threshold 0.4')
    plt.scatter(x4, y4, color='black')
    plt.plot(x5, y5, color='yellow', label='Threshold 0.5')
    plt.scatter(x5, y5, color='yellow')

    plt.title(hop_duration)
    plt.xlabel('Window Duration')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()


def plot_hop_duration_precision(start=0, threshold='0.1'):
    x1 = [work[i][1] for i in range(start, start+20, 5)]
    y1 = [work[i][3] for i in range(start, start+20, 5)]

    x2 = [work[i][1] for i in range(start+20, start+40, 5)]
    y2 = [work[i][3] for i in range(start+20, start+40, 5)]

    x3 = [work[i][1] for i in range(start+40, start+60, 5)]
    y3 = [work[i][3] for i in range(start+40, start+60, 5)]

    x4 = [work[i][1] for i in range(start+60, start+80, 5)]
    y4 = [work[i][3] for i in range(start+60, start+80, 5)]

    plt.plot(x1, y1, color='red', label='Window Duration : 20ms')
    plt.scatter(x1, y1,color='red')
    plt.plot(x2, y2, color='blue', label='Window Duration : 30ms')
    plt.scatter(x2, y2, color='blue')
    plt.plot(x3, y3, color='green', label='Window Duration : 40ms')
    plt.scatter(x3, y3, color='green')
    plt.plot(x4, y4, color='black', label='Window Duration : 50ms')
    plt.scatter(x4, y4, color='black')

    plt.title(threshold)
    plt.xlabel('Hop Duration')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def plot_hop_duration_recall(start=0, threshold='0.1'):
    x1 = [work[i][1] for i in range(start, start+20, 5)]
    y1 = [work[i][4] for i in range(start, start+20, 5)]
    x2 = [work[i][1] for i in range(start+20, start+40, 5)]
    y2 = [work[i][4] for i in range(start+20, start+40, 5)]
    x3 = [work[i][1] for i in range(start+40, start+60, 5)]
    y3 = [work[i][4] for i in range(start+40, start+60, 5)]
    x4 = [work[i][1] for i in range(start+60, start+80, 5)]
    y4 = [work[i][4] for i in range(start+60, start+80, 5)]

    plt.plot(x1, y1, color='red', label='Window Duration : 20ms')
    plt.scatter(x1, y1,color='red')
    plt.plot(x2, y2, color='blue', label='Window Duration : 30ms')
    plt.scatter(x2, y2, color='blue')
    plt.plot(x3, y3, color='green', label='Window Duration : 40ms')
    plt.scatter(x3, y3, color='green')
    plt.plot(x4, y4, color='black', label='Window Duration : 50ms')
    plt.scatter(x4, y4, color='black')

    plt.title(threshold)
    plt.xlabel('Hop Duration')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

plot_threshold_precision(0, '20ms')
plot_threshold_precision(20, '30ms')
plot_threshold_precision(40, '40ms')
plot_threshold_precision(60, '50ms')

plot_threshold_recall(0, '20ms')
plot_threshold_recall(20, '30ms')
plot_threshold_recall(40, '40ms')
plot_threshold_recall(60, '50ms')

plot_window_duration_precision(0, '3ms')
plot_window_duration_precision(5, '5ms')
plot_window_duration_precision(10, '7ms')
plot_window_duration_precision(15, '10ms')

plot_window_duration_recall(0, '3ms')
plot_window_duration_recall(5, '5ms')
plot_window_duration_recall(10, '7ms')
plot_window_duration_recall(15, '10ms')

plot_hop_duration_precision(0, '0.1')
plot_hop_duration_precision(1, '0.2')
plot_hop_duration_precision(2, '0.3')
plot_hop_duration_precision(3, '0.4')
plot_hop_duration_precision(4, '0.5')

plot_hop_duration_recall(0, '0.1')
plot_hop_duration_recall(1, '0.2')
plot_hop_duration_recall(2, '0.3')
plot_hop_duration_recall(3, '0.4')
plot_hop_duration_recall(4, '0.5')

