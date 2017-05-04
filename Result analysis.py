"""
Result analysis.py
Observing results of the vowel elimination algorithm in different time bands. Results stored in file with _Analysis.csv
extension.

Author: Rishabh Brajabasi
Date: 2nd May 2017
"""

file_name_template_1 = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Evaluation PE V6\\Vowel_Evaluation_V6_Test_7.csv'
results_vowel = open(file_name_template_1)  # The csv file where the results are saved
file_name_template_2 = file_name_template_1[:-4] + '_Analysis.csv'
results_analysis = open(file_name_template_2, 'w')  # The csv file where the results are saved
data = results_vowel.read()
one = data.split('\n')
two = []
for i in range(len(one)):
    two.append(one[i].split(','))
two.pop(0)
two.pop(-1)
results_analysis.write('Start time' + ',' + 'End time' + ',' + 'Count' + ',' + 'Precision' + ',' + 'Recall' + ',' + 'Files' + '\n')
def results(analyze, time_1, time_2):
    count = 0
    precision = []
    recall = []
    names = []
    for element in range(len(one) - 2):
        if analyze[element][8] == 'Fine':
            if time_1 < float(analyze[element][5]) < time_2:
                    count += 1
                    precision.append(float(analyze[element][6]))
                    recall.append(float(analyze[element][7]))
                    names.append(analyze[element][0])
    results_analysis.write(str(start_time) + ',' + str(end_time) + ',' + str(count) + ',' + str(sum(precision)/len(precision)) + ',' + str(sum(recall)/len(recall)) + ',' + str(names) + ',' + '\n')

start_time = 0.0
end_time = 0.5
results(two, start_time, end_time)

start_time = 0.5
end_time = 1.0
results(two, start_time, end_time)

start_time = 1.0
end_time = 1.5
results(two, start_time, end_time)

start_time = 1.5
end_time = 2.0
results(two, start_time, end_time)

start_time = 2.0
end_time = 2.5
results(two, start_time, end_time)

start_time = 2.5
end_time = 3.0
results(two, start_time, end_time)

start_time = 3.0
end_time = 3.5
results(two, start_time, end_time)

start_time = 3.5
end_time = 4.0
results(two, start_time, end_time)

start_time = 4.0
end_time = 4.5
results(two, start_time, end_time)

start_time = 4.5
end_time = 5.0
results(two, start_time, end_time)

start_time = 5.0
end_time = 5.5
results(two, start_time, end_time)

start_time = 5.5
end_time = 6.0
results(two, start_time, end_time)

start_time = 6.0
end_time = 6.5
results(two, start_time, end_time)

start_time = 6.5
end_time = 7.0
results(two, start_time, end_time)

start_time = 7.0
end_time = 7.5
results(two, start_time, end_time)

start_time = 7.5
end_time = 8.0
results(two, start_time, end_time)

start_time = 8.0
end_time = 8.5
results(two, start_time, end_time)

start_time = 8.5
end_time = 9.0
results(two, start_time, end_time)

start_time = 9.0
end_time = 9.5
results(two, start_time, end_time)

start_time = 9.5
end_time = 10.0
results(two, start_time, end_time)

start_time = 10.0
end_time = 10.5
results(two, start_time, end_time)

start_time = 10.5
end_time = 11.0
results(two, start_time, end_time)

start_time = 11.0
end_time = 11.5
results(two, start_time, end_time)

start_time = 11.5
end_time = 12.0
results(two, start_time, end_time)

start_time = 12.0
end_time = 60.0
results(two, start_time, end_time)
