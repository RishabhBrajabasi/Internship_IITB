import re

text_grid_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Test_accuracy.TextGrid', 'r')
text_grid_2 = open('F:\Projects\Active Projects\Project Intern_IITB\Convex Hull\R3_Mark.TextGrid', 'r')

data_1 = text_grid_1.read()
data_2 = text_grid_2.read()

time_1 = []
time_2 = []


def count(vowel):
    for vw in re.finditer(vowel, data_1):
        time_1.append(float(
            data_1[vw.start() - 38] + data_1[vw.start() - 37] + data_1[vw.start() - 36] + data_1[vw.start() - 35] + data_1[
                vw.start() - 34]))
        time_1.append(float(
            data_1[vw.start() - 19] + data_1[vw.start() - 18] + data_1[vw.start() - 17] + data_1[vw.start() - 16] + data_1[
                vw.start() - 15]))
    return data_1.count(vowel)



count_2 = data_2.count('"Vowel"')
print "Count of Vowel is: ", count_2
for m in re.finditer('"Vowel"', data_2):
    time_2.append(float(
        data_2[m.start() - 34] + data_2[m.start() - 33] + data_2[m.start() - 32] + data_2[m.start() - 31] + data_2[
            m.start() - 30] + data_2[m.start() - 29]))
    time_2.append(float(
        data_2[m.start()-17] + data_2[m.start() - 16] + data_2[m.start() - 15] + data_2[m.start() - 14] + data_2[m.start() - 13] + data_2[
            m.start() - 12]))


c1 = count('"aa"')
print time_1
c2 = count('"AA"')
print time_1
c3 = count('"ae"')
print time_1
c4 = count('"aw"')
print time_1
c5 = count('"ay"')
print time_1
c6 = count('"ee"')
print time_1
c7 = count('"ex"')
print time_1
c8 = count('"ii"')
print time_1
c9 = count('"II"')
print time_1
c10 = count('"oo"')
print time_1
c11 = count('"OO"')
print time_1
c12 = count('"oy"')
print time_1
c13 = count('"uu"')
print time_1
c14 = count('"UU"')
print time_1

c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14
print "Vowels FA : ", c
# print time_1
# print time_2

counter = 0
#
# for j in range(0, len(time_2), 2):
#     for i in range(0, len(time_1)+1, 2):
#         if time_1[j] < time_2[i] < time_1[j+1]:
#             print time_1[j], '<', time_2[i], '<', time_1[j+1]
#

count_element_time = []

for j in range(1, len(time_2), 2):
    for i in range(0, len(time_1), 2):
        if time_1[i] < time_2[j] < time_1[i+1] and time_2[j-1] < time_1[i] < time_1[i+1]:
            # print time_1[i], '<', time_2[j], '<', time_1[i+1]
            counter += 1
            count_element_time.append(time_2[j-1])
            count_element_time.append(time_2[j])
# print "\n"

for j in range(0, len(time_2), 2):
    for i in range(0, len(time_1), 2):
        if time_1[i] < time_2[j] < time_1[i+1] and time_1[i] < time_1[i+1] < time_2[j+1]:
            # print time_1[i], '<', time_2[j], '<', time_1[i+1]
            counter += 1
            if time_2[j] not in count_element_time:
                count_element_time.append(time_2[j])
                count_element_time.append(time_2[j+1])

for j in range(0, len(time_2), 2):
    for i in range(0, len(time_1), 2):
        if time_1[i] < time_2[j] < time_1[i+1] and time_1[i] < time_2[j + 1] < time_1[i+1]:
            counter += 1
            if time_2[j] not in count_element_time:
                count_element_time.append(time_2[j])
                count_element_time.append(time_2[j+1])

#
# print "\n"
# print counter

# print count_element_time
print len(count_element_time)/2
