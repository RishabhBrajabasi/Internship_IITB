import matplotlib.pyplot as plt

precision = [80.1043244,75.0974532,87.4238889,67.409107,82.0403132,62.8730325,79.3278255]
recall = [76.0498987,83.6897131,75.3561366,85.8581758,79.1702267,90.2118614,84.9815972]
label = ['M0','M1','M2','M3','M4','M5','M6']

plt.scatter(recall,precision)
for j in range(len(label)):
    plt.text(recall[j],precision[j],label[j])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.vlines(90,0,100)
plt.hlines(90,0,100)

plt.vlines(80,0,100)
plt.hlines(80,0,100)

plt.xlim(0,100)
plt.ylim(0,100)
plt.show()
