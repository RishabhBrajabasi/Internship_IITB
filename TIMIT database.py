import fnmatch
import os
from shutil import copyfile

matches = []
for root, dirnames, filenames in os.walk('F:\Projects\Active Projects\Project Intern_IITB\\timit\\cd1_timit\\train'):
    for filename in fnmatch.filter(filenames, '*.wav'):
        matches.append(os.path.join(root, filename))
for i in range(len(matches)):
    start = matches[i].find('test')
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+5:-4].replace('\\', '_') + '_TRAIN.wav')

matches = []
for root, dirnames, filenames in os.walk('F:\Projects\Active Projects\Project Intern_IITB\\timit\\cd1_timit\\train'):
    for filename in fnmatch.filter(filenames, '*.phn'):
        matches.append(os.path.join(root, filename))
for i in range(len(matches)):
    start = matches[i].find('test')
    copyfile(matches[i], 'F:\Projects\Active Projects\Project Intern_IITB\\timit\\TIMIT Database\\' + matches[i][start+5:-4].replace('\\', '_') + '_TRAIN.phn')
