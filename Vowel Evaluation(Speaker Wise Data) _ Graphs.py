import matplotlib.pyplot as plt

speaker_results = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation (Speaker Wise Data)\\Vowel_Evaluation_V5_Speaker_Based.csv', 'r')
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

for j in range(len(data)):
    name.append(data[j][0])
    precision.append(data[j][1])
    recall.append(float(data[j][2]))
    no_of_files.append(data[j][3])

axis_p = []
axis_r = []
for j in range(len(name)):
    axis_p.append(j)
    axis_r.append(j+0.1)


# plt.stem(axis_p,precision,'red',label='Precision')
# plt.stem(axis_r,recall,'blue',label='Recall')
plt.scatter(axis_p,precision,color='red',label='Precision')
plt.scatter(axis_r,recall,color='blue',label='Recall')
for j in range(len(axis_p)):
    plt.vlines(axis_p[j],0,precision[j])
for j in range(len(axis_r)):
    plt.vlines(axis_r[j],0,recall[j])
plt.xlabel('Speaker No')
plt.ylabel('Precision and Recall')
for j in range(len(axis_p)):
    plt.text(axis_p[j]+0.01, recall[j]+0.01, str(no_of_files[j]),fontsize='10')
plt.xlim(-0.5,len(axis_r)+0.5)
plt.ylim(0,1.1)
plt.hlines(0.8,0,len(axis_r))
# plt.legend()
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show()