import glob
from shutil import copyfile

file_name_template_1 = 'F:\Projects\Active Projects\Project G\Ruchita\\'

file_no = 1
# results_vowel = open(file_name_template_1 + result_name, 'w')  # The csv file where the results are saved
only_audio = glob.glob(file_name_template_1 + 'Memes\\*.jpg')  # Extract file name of all audio samples
# only_text = glob.glob(file_name_template_1 + '\Data 1\*.TextGrid')  # Extract file name of all force aligned TextGrid

print only_audio

for j in only_audio:
    copyfile(j, file_name_template_1 + "Sorted\\" + str(file_no) + '.jpg')
    file_no += 1
