"""
TIMIT database.py
Purpose of this script is to extract the files .wav files and .phn files from the various folders and sub-folders in the
TIMIT database and place then in a separate folder.

Author: Rishabh Brajabasi
Date: 24th April 2017
"""


import fnmatch
import os
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
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+5:-4].replace('\\', '_') + '_T2.wav')

matches = []
for root, dirnames, filenames in os.walk('F:\Projects\Active Projects\Project Intern_IITB\\timit\\cd1_timit\\train'):
    for filename in fnmatch.filter(filenames, '*.phn'):
        matches.append(os.path.join(root, filename))
for i in range(len(matches)):
    start = matches[i].find('train')
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+5:-4].replace('\\', '_') + '_T2.phn')

