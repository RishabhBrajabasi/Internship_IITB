import glob
from shutil import copyfile

clean = open('F:\Projects\Active Projects\Project Intern_IITB\ids_with_no_inc_dis.txt')  # List of clean files
data =  clean.read()  # Read the data as a single string
clean_list = data.split('\n')  # Create list, where each element as a file name
only_audio = glob.glob('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Original Data\\child_wav_files\\*.wav')  # Creating list of all Audio files
only_text = glob.glob('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Original Data\GT_TGs_phones\\*.TextGrid')  # Creating list of all TextGrid files

for j in only_audio:
    x = j.find('child_wav_files')
    a_file = j[x+16:-4]
    if a_file in clean_list:
        copyfile(j, 'F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Clean Data\child_wav_files\\' + a_file + '.wav')  # Copying the clean files to a New folder

for j in only_text:
    x = j.find('GT_TGs_phones')
    a_file = j[x+14:-9]
    if a_file in clean_list:
        copyfile(j, 'F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Clean Data\GT_TGs_phones\\' + a_file + '.TextGrid')  # Copying the clean files to a New folder
