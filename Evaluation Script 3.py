import re

text_grid_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Analysis\A2\\06003shaktikhadke_56580b937e63f5035c002739_575525a79ee20a028cfbbc54_8_31111\\06003shaktikhadke_56580b937e63f5035c002739_575525a79ee20a028cfbbc54_8_31111.TextGrid', 'r')
text_grid_2 = open('F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Analysis\A2\\06003shaktikhadke_56580b937e63f5035c002739_575525a79ee20a028cfbbc54_8_31111\\06003shaktikhadke_56580b937e63f5035c002739_575525a79ee20a028cfbbc54_8_31111PE_NEW.TextGrid', 'r')

data_1 = text_grid_1.read()
data_2 = text_grid_2.read()

time_1 = []
time_2 = []

counter = 0
for m in re.finditer('text = "', data_1):
    if data_1[m.start() - 33] == '=':
        time_1.append(float(
            data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] + data_1[m.start() - 29] + data_1[m.start() - 28] + data_1[
                m.start() - 27] + data_1[m.start() - 26]))
        time_1.append(float(
            data_1[m.start() - 13] + data_1[m.start()-12] + data_1[m.start() - 11] + data_1[m.start() - 10] + data_1[m.start() - 9] + data_1[m.start() - 8] + data_1[
                m.start() - 7] + data_1[m.start() - 6] + data_1[m.start() - 5]))
    else:
        time_1.append(float(
            data_1[m.start() - 33] + data_1[m.start() - 32] + data_1[m.start() - 31] + data_1[m.start() - 30] + data_1[m.start() - 29] + data_1[
                m.start() - 28] + data_1[m.start() - 27] + data_1[m.start() - 26]))
        time_1.append(float(
            data_1[m.start() - 13] + data_1[m.start() - 12] + data_1[m.start() - 11] + data_1[m.start() - 10] + data_1[
                m.start() - 9] + data_1[m.start() - 8] + data_1[
                m.start() - 7] + data_1[m.start() - 6] + data_1[m.start() - 5]))

    if data_1[m.start() + 9] == '"':
        time_1.append(data_1[m.start() + 8])
    elif data_1[m.start() + 10] == '"':
        time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9])
    else:
        time_1.append(data_1[m.start() + 8] + data_1[m.start() + 9] + data_1[m.start() + 10])

    time_1.append(counter)
    counter += 1


for m in re.finditer('"Vowel"', data_2):
    time_2.append(float(
        data_2[m.start() - 34] + data_2[m.start() - 33] + data_2[m.start() - 32] + data_2[m.start() - 31] + data_2[
            m.start() - 30] + data_2[m.start() - 29]))
    time_2.append(float(
        data_2[m.start()-17] + data_2[m.start() - 16] + data_2[m.start() - 15] + data_2[m.start() - 14] + data_2[m.start() - 13] + data_2[
            m.start() - 12]))


def count(vowel):
    for vw in re.finditer(vowel, data_1):
        time_1.append(float(
            data_1[vw.start() - 38] + data_1[vw.start() - 37] + data_1[vw.start() - 36] + data_1[vw.start() - 35] + data_1[
                vw.start() - 34]))
        time_1.append(float(
            data_1[vw.start() - 19] + data_1[vw.start() - 18] + data_1[vw.start() - 17] + data_1[vw.start() - 16] + data_1[
                vw.start() - 15]))
    return data_1.count(vowel)


listing = []

print 't1', time_1
print 't2', time_2

for j in range(0, len(time_2), 2):
    for i in range(0, len(time_1), 4):
        # print 'Out', time_1[i], time_1[i + 1]
        if time_1[i] <= time_2[j] < time_1[i+1] and time_1[i] < time_2[j+1] <= time_1[i+1]:
            listing.append(time_1[i+2])
            listing.append(time_1[i+3])
        # if time_1[i] <= time_2[j+2] < time_1[i + 1] and time_1[i] < time_2[j + 3] <= time_1[i + 1]:
        #     listing.append(time_1[i + 2])

for j in range(0, len(time_2), 2):
    for i in range(0, len(time_1) - 4, 4):
        print time_1[i], time_1[i+1], time_1[i+4], time_1[i+5]
        if time_1[i] < time_2[j] < time_1[i+1] and time_1[i+4] < time_2[j+1] < time_1[i+5]:
            # if time_1[i+1] - time_2[j] > time_2[j+1] - time_1[i+3]:
            listing.append(time_1[i+3])
            listing.append(time_1[i+4])
            # else:
            listing.append(time_1[i+6])
            listing.append(time_1[i+7])
            print time_1[i+3],time_1[i+4],time_1[i+6],time_1[i+7]


            # first = time_1[i + 1] - time_2[j]
            # second = time_2[j + 1] - time_1[i + 3]
            # overall = time_2[j + 1] - time_2[j]

# print listing
count_2 = data_2.count('"Vowel"')
print "Count of Vowel according to one of algo's is: ", count_2


c1 = count('"aa"')
c2 = count('"AA"')
c3 = count('"ae"')
c4 = count('"aw"')
c5 = count('"ay"')
c6 = count('"ee"')
c7 = count('"ex"')
c8 = count('"ii"')
c9 = count('"II"')
c10 = count('"oo"')
c11 = count('"OO"')
c12 = count('"oy"')
c13 = count('"uu"')
c14 = count('"UU"')

c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14
print "Count of vowel according to FA TextGrid is : ", c

count = 0
vowel_data = ['aa', 'AA', 'ae', 'aw', 'ay', 'ee', 'ex', 'ii', 'II', 'oo', 'OO', 'oy', 'uu', 'UU']

print listing

already_here = []
for vowel_sound in range(0, len(listing), 2):
    if listing[vowel_sound] in vowel_data and listing[vowel_sound + 1] not in already_here:
        count += 1
        already_here.append(listing[vowel_sound + 1])

print "No of vowel coincident are", count
