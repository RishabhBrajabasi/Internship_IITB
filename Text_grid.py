#!/usr/bin/python

# ---------------------------------------------------------------------------------
''' This script can be used for preparing a TextGrid file from the csv output file
of any other program. It takes as input the name of the csv file and generates a
TextGrid file with the same name. The format of the csv file should be as follows:

startTime1	endTime1	label1
startTime2	endTime2	label2
startTime3	endTime3	label3
.
.
.

NOTE:	1. This script must be kept in the same folder as the input csv file.
	2. The field separator used in the csv file must be a "\t" (tab).

Input argument: 1. Name of csv file (with .csv extension)
Output : A TextGrid file (which can be viewed in Praat) with intervals marked
	 and labelled as per the data in the csv file.

Author: Prakhar Swarup
Last updated: 17 Jan 2017
'''
# --------------------------------------------------------------------------------

import sys
import os
import csv

csvFileName = 'F:\Projects\Active Projects\Project Intern_IITB\Segmentation\Segments.csv'  # Provide csv file name as input (Input argument)

TGFileName = csvFileName.split('.')[0] + '.TextGrid'  # Setting name of TextGrid file

fidcsv = open(csvFileName, 'r')
fidTG = open(TGFileName, 'w')

reader = csv.reader(fidcsv, delimiter="\t")  # Reading data from csv file
data = list(reader)  # Converting read data into python list format
label_count = len(data)  # Finding total number of rows in csv file
end_time = data[-1][1]

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
    fidTG.write('\t\t\txmin = ' + str(data[j][0]) + '\n')
    fidTG.write('\t\t\txmax = ' + str(data[j][1]) + '\n')
    fidTG.write('\t\t\ttext = "' + str(data[j][2]) + '"\n')

fidcsv.close()
fidTG.close()
