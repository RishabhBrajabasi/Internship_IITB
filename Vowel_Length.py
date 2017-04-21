import re
from os import listdir

from os.path import isfile, join


onlyfiles = [f for f in listdir('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\GT_TGs_phones') if isfile(join('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\GT_TGs_phones', f))]

results_vowel = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Results_Vowel.csv', 'a')
results_overall = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Results_overall.csv', 'a')

results_vowel.write('File Name' + ',' + 'No. of times Vowel ''a'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''AA'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''ae'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''aw'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''ay'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''ee'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''ex'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''ii'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''II'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''oo'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''OO'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''oy'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''uu'' occurred' + ',' + 'Avg. Duration of Vowel' +
                    ',' + 'No. of times Vowel ''UU'' occurred' + ',' + 'Avg. Duration of Vowel' + '\n')

results_overall.write('File Name' + ',' + 'No. of Vowels' + ',' + 'Avg Duration of Vowel' + '\n')

for j in onlyfiles:
    text_grid_1 = open('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\GT_TGs_phones\\' + j, 'r')

    data_1 = text_grid_1.read()
    time_1 = []
    len_vw = []


    def count(vowel):
        time_1[:] = []
        for vw in re.finditer(vowel, data_1):
            time_1.append(float(
                data_1[vw.start() - 38] + data_1[vw.start() - 37] + data_1[vw.start() - 36] + data_1[vw.start() - 35] +
                data_1[
                    vw.start() - 34]))
            time_1.append(float(
                data_1[vw.start() - 19] + data_1[vw.start() - 18] + data_1[vw.start() - 17] + data_1[vw.start() - 16] +
                data_1[
                    vw.start() - 15]))
        return data_1.count(vowel), time_1


    def length(list_time):
        len_vw[:] = []
        if len(list_time) != 0:
            for vlen in range(1, len(list_time), 2):
                len_vw.append(list_time[vlen] - list_time[vlen - 1])
            # print sum(len_vw) / len(len_vw), '\n'
            return sum(len_vw) / len(len_vw)
        else:
            # print 'Vowel not present in audio\n'
            return 0


    # print 'a'
    c1, time_a = count('"aa"')
    v1 = length(time_a)

    # print "aa"
    c2, time_AA = count('"AA"')
    v2 = length(time_AA)

    # print "ae"
    c3, time_ae = count('"ae"')
    v3 = length(time_ae)

    # print "aw"
    c4, time_aw = count('"aw"')
    v4 = length(time_aw)

    # print "ay"
    c5, time_ay = count('"ay"')
    v5 = length(time_ay)

    # print "ee"
    c6, time_ee = count('"ee"')
    v6 = length(time_ee)

    # print "ex"
    c7, time_ex = count('"ex"')
    v7 = length(time_ex)

    # print "ii"
    c8, time_ii = count('"ii"')
    v8 = length(time_ii)

    # print "II"
    c9, time_II = count('"II"')
    v9 = length(time_II)

    # print "oo"
    c10, time_oo = count('"oo"')
    v10 = length(time_oo)

    # print "OO"
    c11, time_OO = count('"OO"')
    v11 = length(time_OO)

    # print "oy"
    c12, time_oy = count('"oy"')
    v12 = length(time_oy)

    # print "uu"
    c13, time_uu = count('"uu"')
    v13 = length(time_uu)

    # print "UU"
    c14, time_UU = count('"UU"')
    v14 = length(time_UU)

    c = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14
    v = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13 + v14

    results_vowel.write(j + ',' + str(c1) + ',' + str(v1) + ',' + str(c2) + ',' + str(v2) + ',' + str(c3) + ',' + str(v3)
                          + ',' + str(c4) + ',' + str(v4) + ',' + str(c5) + ',' + str(v5) + ',' + str(c6) + ',' + str(v6)
                          + ',' + str(c7) + ',' + str(v7) + ',' + str(c8) + ',' + str(v8) + ',' + str(c9) + ',' + str(v9)
                          + ',' + str(c10) + ',' + str(v10) + ',' + str(c11) + ',' + str(v11) + ',' + str(c12)
                          + ',' + str(v12) + ',' + str(c13) + ',' + str(v13) + ',' + str(c14) + ',' + str(v14) + '\n')


    results_overall.write(j + ',' + str(c) + ',' + str(v) + '\n')



    # print "Vowels FA : ", c
    # print "Vowel Length : ", v / c

results_vowel.close()
results_overall.close()
