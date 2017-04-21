import glob
import os
from shutil import copyfile

file_name_template_1 = "F:\Projects\Active Projects\Project Intern_IITB\Speaker Wise Data\\"


audio_file = glob.glob(file_name_template_1 + 'Data 1\*.wav')  # Extract file name of all audio samples
text_grid_file = glob.glob(file_name_template_1 + '\Data 1\*.TextGrid')  # Extract file name of all force aligned TextGrid

only_audio = []
for audio in audio_file:
    start = str(audio).find('Data 1')
    only_audio.append(audio[start+8:])
file_name = []
for audio in only_audio:
    stop = str(audio).find('_')
    file_name.append(audio[:stop])
u_file_name = []
for j in file_name:
    if j not in u_file_name:
        u_file_name.append(j)
for k in u_file_name:
    if not os.path.exists(file_name_template_1 + "Speaker_" + str(k) + "\\Data 1"):
        os.makedirs(file_name_template_1 + "Speaker_" + str(k) + "\\Data 1")
    if not os.path.exists(file_name_template_1 + "Speaker_" + str(k) + "\\Data 2"):
        os.makedirs(file_name_template_1 + "Speaker_" + str(k) + "\\Data 2")

for j in audio_file:
    for k in u_file_name:
        if str(j).find(str(k)) != -1:
            start = str(j).find('Data 1')
            copyfile(j,file_name_template_1 + "Speaker_" + str(k) + "\\Data 1\\" + str(j[start+8:]))
            copyfile(j[:-4]+'.TextGrid',file_name_template_1 + "Speaker_" + str(k) + "\\Data 1\\" + str(j[start+8:-4] + '.TextGrid'))
