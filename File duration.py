from os import listdir
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

onlyfiles = [f for f in listdir('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\GT_TGs_phones') if isfile(join('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\GT_TGs_phones', f))]
# Length = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\LengthFinal.csv', 'a')
distribution_1 = 0
distribution_2 = 0
distribution_3 = 0
distribution_4 = 0
distribution_5 = 0
distribution_6 = 0
distribution_7 = 0
distribution_8 = 0
distribution_9 = 0
# Length.write('File Name' + ',' + 'Start Time' + ',' + 'Stop Time' + ',' + 'Duration' + '\n')
for j in onlyfiles:
    text_grid_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\GT_TGs_phones\\' + j, 'r')
    data_1 = text_grid_1.read()
    start = data_1.find('xmin')
    stop = data_1.find('xmax')
    starting = float(data_1[start + 7])
    stopping = float(data_1[stop + 7] + data_1[stop + 8] + data_1[stop + 9] + data_1[stop + 10] + data_1[stop + 11] + data_1[stop + 12])
    duration = stopping - starting
    if duration < 1.0:
        distribution_1 += 1
    elif 1.0 <= duration < 2.0:
        distribution_2 += 1
    elif 2.0 <= duration <= 3.0:
        distribution_3 += 1
    elif 3.0 <= duration <= 4.0:
        distribution_4 += 1
    elif 4.0 <= duration <= 5.0:
        distribution_5 += 1
    elif 5.0 <= duration <= 6.0:
        distribution_6 += 1
    elif 6.0 <= duration <= 7.0:
        distribution_7 += 1
    elif 7.0 <= duration <= 8.0:
        distribution_8 += 1
    else:
        distribution_9 += 1
    # Length.write(j + ',' + str(starting) + ',' + str(stopping) + ',' + str(duration) + '\n')


fig = plt.figure()
ax = fig.add_subplot(111)

y = [distribution_1, distribution_2, distribution_3, distribution_4, distribution_5, distribution_6, distribution_7, distribution_8, distribution_9]

N = len(y)
ind = np.arange(N)
x = range(N)
x1 = ['Dur < 1', '1 < Dur < 2', '2 < Dur < 3', '3 < Dur < 4', '4 < Dur < 5', '5 < Dur < 6', '6 < Dur < 7', '7 < Dur < 8', 'Dur > 8']
width = 1/1.5
plt.bar(x, y, width=width)

labels = ["label%d" % i for i in xrange(len(x))]

for label in range(len(y)):
    plt.text(x[label] + 0.4, y[label], y[label], ha='center', va='bottom')



ax.set_xticks(ind + 0.3)
xtickNames = ax.set_xticklabels(x1)
plt.setp(xtickNames, rotation=0, fontsize=10)

plt.ylabel('No. of samples')
plt.title('Distribution of time duration of utterances')
plt.show()

# Length.close()

