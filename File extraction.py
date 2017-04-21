from shutil import copyfile
import os
from datetime import datetime

startTime = datetime.now()  # To calculate the run time of the code.

analyse = open('C:\Users\Mahe\Desktop\\hello.txt', 'r')  # CHANGE THIS

results = analyse.read()

print results
#
# words = results.split("\n")
# path_of_origin = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Data 1'
# path_of_inquiry = 'F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Data 2'
# path_of_delivery = "F:\Projects\Active Projects\Project Intern_IITB\Vowel Comparison V1\Analysis\All_of_it"  # CHANGE THIS
#
#
# for j in range(len(words) - 1):
#
#     os.makedirs(path_of_delivery + '\\' + str(j+1))
#
#
# for j in range(len(words) - 1):
#
#     src = path_of_inquiry + words[j][74:-4] + 'CH_NEW.TextGrid'
#     dst = path_of_delivery + '\\' + str(j+1) + '\\' + words[j][74:-4] + 'CH_NEW.TextGrid'
#     copyfile(src, dst)
#
#     src = path_of_inquiry + words[j][74:-4] + 'PE_NEW.TextGrid'
#     dst = path_of_delivery + '\\' + str(j+1) + '\\' + words[j][74:-4] + 'PE_NEW.TextGrid'
#     copyfile(src, dst)
#
#     src = path_of_origin + words[j][74:-4] + '.TextGrid'
#     dst = path_of_delivery + '\\' + str(j+1) + '\\' + words[j][74:-4] + '.TextGrid'
#     copyfile(src, dst)
#
#     src = path_of_origin + words[j][74:]
#     dst = path_of_delivery + '\\' + str(j+1) + '\\' + words[j][74:]
#     copyfile(src, dst)
#
# print datetime.now() - startTime  # Print program run time
#
