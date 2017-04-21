import glob
import shutil

only_audio = glob.glob('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Mixed\*.wav')
only_text = glob.glob('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Mixed\*.TextGrid')

only_audio_name = []
only_text_name = []
count = 0

for j in only_audio:
    only_audio_name.append(j[: -4])

for j in only_text:
    only_text_name.append(j[: -9])

for j in only_audio_name:
    if j in only_text_name:
        count += 1
        shutil.copy2(str(j) + '.wav', 'F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Newmix\\' + str(j[71:]) + '.wav')
        shutil.copy2(str(j) + '.TextGrid', 'F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Newmix\\' + str(j[71:]) + '.TextGrid')
        # copyfile('F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Mixed' + j + '.wav',
        #          'F:\Projects\Active Projects\Project Intern_IITB\Rishabh_FA_Audio\Newmix' + j + '.wav')


print count
print len(only_audio_name)
print len(only_text_name)
