'''Sta arxeia prepei prwta na exei trexei to script file_extractor.py etsi wste na einai se morfh id.tmp '''
import os
import csv
from definitions import AUDIO_DIR

if __name__ == '__main__':
    filenames = os.listdir(AUDIO_DIR)
    with open('Survey.csv', 'rb') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for idx, filename in enumerate(filenames):
                if row['healthCode'] == filename[:-5]:
                    os.rename(AUDIO_DIR + '/' + filename, AUDIO_DIR + '/'
                              + filename[:-5] + '_' + row['professional-diagnosis'] + '.alac')
