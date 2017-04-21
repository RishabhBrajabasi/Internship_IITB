import matplotlib.pyplot as plt

speaker_results = open('F:\Projects\Active Projects\Project Intern_IITB\Desktop\\Vowel_opt_V3_MA.csv', 'r')
sr = speaker_results.read()
# print sr
list_data = sr.split('\n')
# print list_data
list_data.pop(0)
list_data.pop(-1)
list_data.pop(-1)
# print list_data
data = []
for j in list_data:
    data.append((j.split(',')))

name = []
precision = []
recall = []
no_of_files =[]
precision_rouge = []
recall_rouge = []


for j in range(len(data)):
    name.append(data[j][0])
    precision.append(data[j][10])
    recall.append(data[j][11])
    precision_rouge.append(data[j][10])
    # recall_rouge.append(data[j][11])
    # no_of_files.append(data[j][3])
#
precision_rouge.sort()
for j in range(len(precision_rouge)):
    recall_rouge.append(recall[precision.index(precision_rouge[j])])


# recall_rouge.sort()
# for j in range(len(recall_rouge)):
#     precision_rouge.append(precision[recall.index(recall_rouge[j])])


axis_p = []
axis_r = []
for j in range(len(name)):
    axis_p.append(j)
    axis_r.append(j)

plt.scatter(axis_p,precision_rouge,color='red',label='Precision')
plt.scatter(axis_r,recall_rouge,color='blue',label='Recall')
for j in range(len(axis_p)):
    plt.vlines(axis_p[j],0,precision_rouge[j],colors='black')
for j in range(len(axis_r)):
    plt.vlines(axis_r[j],0,recall_rouge[j],colors='black')
plt.xlim(-0.5,len(axis_r)+0.5)
plt.ylim(0,1)
plt.grid()
plt.xlabel('Version No')
plt.ylabel('Precision and Recall')
plt.hlines(0.8,0,len(axis_r))
plt.hlines(0.9,0,len(axis_r))
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show()