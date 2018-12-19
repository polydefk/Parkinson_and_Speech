'''Prepei na einai exei trexei to script file_transformer.py dhladh ta arxeia na einai se morfh 5827528_idont_take...._audio.wav gia na doulepsei
   Prepei epishs na vriskontai ston fakelo Files '''
import os
import sys
from definitions import AUDIO_DIR
import shutil

if __name__ == '__main__':
    if not os.path.exists(AUDIO_DIR + '/PD'):
        os.mkdir(AUDIO_DIR + '/PD')
    if not os.path.exists(AUDIO_DIR + '/HC'):
        os.mkdir(AUDIO_DIR + '/HC')
    filenames = os.listdir(AUDIO_DIR)
    for idx, filename in enumerate(filenames):
        if filename[37:-5] == 'true' and not(dir(filename)):
            shutil.move(AUDIO_DIR + '/' + filename, AUDIO_DIR + '/PD')
        else :
            shutil.move(AUDIO_DIR + '/' + filename, AUDIO_DIR + '/HC')