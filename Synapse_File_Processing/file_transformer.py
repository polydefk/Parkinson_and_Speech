'''Sta arxeia prepei prwta na exei trexei to script file_organizerr.py etsi wste na einai se morfh id_bla_bla.m4a '''
import os
import sys
import subprocess
import shutil
from definitions import AUDIO_DIR

# <--- to directory poy trexei to script

working_directory = AUDIO_DIR
os.chdir(working_directory)


def main():
    filenames = []  # <------ pinakas me ola ta filename sto working directory poy teleiwnoun se .m4a
    for filename in os.listdir(working_directory):
        if filename.endswith('.m4a'):
            filenames.append(filename)

    for filename in filenames:
        subprocess.call([
            "ffmpeg", "-i",
            os.path.join(working_directory, filename),
            "-ar", "16000",
            os.path.join(working_directory, '%s.wav' % filename[:-4])
            # <====== metatroph toy m4a arxeiou se wav kai rixnw to samplerate sta 16kHz
        ])

    filenames = []
    for filename in os.listdir(working_directory):  # <======== xana ftiaxnw ena pinaka apo ta ftiagmena wav pleon
        if filename.endswith('.wav'):
            filenames.append(filename)

    for filename in filenames:  # <====== stelnw ta arxeia ston kainourio directory
        if filename.endswith('.wav'):
            old_path = os.path.join(working_directory, filename)
            new_path = []
            print (old_path)
            print (new_path)
            os.rename(old_path, new_path)

    shutil.rmtree(working_directory)
    return 0


if __name__ == '__main__':
    sub_dirs = os.listdir(AUDIO_DIR)
    for idx1, sub_dir in enumerate(sub_dirs):
        filenames = os.listdir(sub_dir)
        print (len(filenames))
        for idx2, filename in enumerate(filenames):
            full_path =  AUDIO_DIR + '/' + sub_dir + '/' + filename
