"""
TIMIT database.py
Purpose of this script is to extract the files .wav files and .phn files from the various folders and sub-folders in the
TIMIT database and place then in a separate folder.

Author: Rishabh Brajabasi
Date: 24th April 2017
"""


import fnmatch
import os
import glob
import csv
from shutil import copyfile

matches = []
for root, dirnames, filenames in os.walk('F:\Projects\Active Projects\Project Intern_IITB\\timit\\cd1_timit\\test'):
    for filename in fnmatch.filter(filenames, '*.wav'):
        matches.append(os.path.join(root, filename))
for i in range(len(matches)):
    start = matches[i].find('test')
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+5:-4].replace('\\', '_') + '_T1.wav')

matches = []
for root, dirnames, filenames in os.walk('F:\Projects\Active Projects\Project Intern_IITB\\timit\\cd1_timit\\test'):
    for filename in fnmatch.filter(filenames, '*.phn'):
        matches.append(os.path.join(root, filename))
for i in range(len(matches)):
    start = matches[i].find('test')
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+5:-4].replace('\\', '_') + '_T1.phn')

matches = []
for root, dirnames, filenames in os.walk('F:\Projects\Active Projects\Project Intern_IITB\\timit\\cd1_timit\\train'):
    for filename in fnmatch.filter(filenames, '*.wav'):
        matches.append(os.path.join(root, filename))
for i in range(len(matches)):
    start = matches[i].find('train')
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+6:-4].replace('\\', '_') + '_T2.wav')

matches = []
for root, dirnames, filenames in os.walk('F:\Projects\Active Projects\Project Intern_IITB\\timit\\cd1_timit\\train'):
    for filename in fnmatch.filter(filenames, '*.phn'):
        matches.append(os.path.join(root, filename))
for i in range(len(matches)):
    start = matches[i].find('train')
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+6:-4].replace('\\', '_') + '_T2.phn')

only_phn = glob.glob('F:\Projects\Active Projects\Project Intern_IITB\\timit\TIMIT Database\*.phn')  # Extract file name of all audio samples

for phn in only_phn:
    csvFileName = phn[:-3] + str('csv')
    text_file_1 = open(csvFileName, 'w')  # Opening CSV file to store results and to create TextGrid

    phone = open(phn, 'r')
    phones = phone.read()
    aa = phones.split('\n')
    for i in range(len(aa)-1):
        split = aa[i].split(' ')
        text_file_1.write('%06.3f' % (float(split[0])*0.0000625) + "\t" + '%06.3f' % (float(split[1])*0.0000625) + "\t" + split[2] + "\n")
    text_file_1.close()

    TGFileName = csvFileName.split('.')[0] + '.TextGrid'  # Setting name of TextGrid file
    fidcsv = open(csvFileName, 'r')
    fidTG = open(TGFileName, 'w')

    reader = csv.reader(fidcsv, delimiter="\t")  # Reading data from csv file
    data_tg = list(reader)  # Converting read data into python list format
    label_count = len(data_tg)  # Finding total number of rows in csv file
    end_time = data_tg[-1][1]

    fidTG.write('File type = "ooTextFile"\n')
    fidTG.write('Object class = "TextGrid"\n')
    fidTG.write('xmin = 0\n')
    fidTG.write('xmax = ' + str(end_time) + '\n')
    fidTG.write('tiers? <exists>\n')
    fidTG.write('size = 1\n')
    fidTG.write('item []:\n')
    fidTG.write('\titem [1]:\n')
    fidTG.write('\t\tclass = "IntervalTier"\n')
    fidTG.write('\t\tname = "Labels"\n')
    fidTG.write('\t\txmin = 0\n')
    fidTG.write('\t\txmax = ' + str(end_time) + '\n')
    fidTG.write('\t\tintervals: size = ' + str(label_count) + '\n');

    for j in range(label_count):
        fidTG.write('\t\tintervals [' + str(j) + ']:\n')
        fidTG.write('\t\t\txmin = ' + str(data_tg[j][0]) + '\n')
        fidTG.write('\t\t\txmax = ' + str(data_tg[j][1]) + '\n')
        fidTG.write('\t\t\ttext = "' + str(data_tg[j][2]) + '"\n')

    fidcsv.close()
    fidTG.close()
