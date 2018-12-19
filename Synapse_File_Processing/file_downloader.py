import shutil
import synapseclient
import numpy as np
from definitions import SYNAPSE_DIR
import csv
import os
import time

if __name__ == '__main__':

    Download_Dir_Full_Path = "/home/polaras/Diploma_PD/newFiles/"
    Csv_Full_Path = SYNAPSE_DIR + "/Survey.csv"  # Demographic Survey csv file
    synapseclient.cache.CACHE_ROOT_DIR = Download_Dir_Full_Path  # set the download directory

    """Credentials"""
    username = ''
    password = ''

    syn = synapseclient.Synapse()
    syn.login(username, password)

    tableID = "syn5511444"
    column = ["audio_audio.m4a"]

    healthID = []
    with open(Csv_Full_Path, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            healthID.append(row[3])
        healthID = np.asarray(healthID)
        healthID = healthID[1:len(healthID)]
    start = time.time()
    for id in healthID:

        if not os.path.exists(Download_Dir_Full_Path+id+'.alac'):
            result = syn.tableQuery("SELECT 'audio_audio.m4a' "
                                    "FROM syn5511444 WHERE healthCode = '%s' LIMIT 2" % id)

            file = syn.downloadTableColumns(result, column)
            try:
                item = file.items()
                print item[0][9] , item[1][0]
                new_name = os.path.join(Download_Dir_Full_Path, id + '.alac')
                shutil.move(item[0][1], new_name)
            except (ValueError, IndexError):
                print("Empty Directory")
        else:
            print("File {0}.alac already exists. .i. \nTime up until now : {1}".format(id,time.time()-start))
    print("Final script time: {0}".format(time.time()-start))
