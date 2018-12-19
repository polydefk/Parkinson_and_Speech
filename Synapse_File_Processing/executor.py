import os
import sys
import subprocess

directory = '/home/polaras/Diploma_PD/code/file_processing'
os.chdir(directory)
''' 1)Bgazw ola ta arxeia apo toys fakelous me to arxeio kwdika file_extractor.py
	2)Metaonomazw ola ta arxeia symfwna me to csv me to arxeio kwdika file_organizer.py
	3)Metatrepw ola ta arxeia apo aac se wav kai ta rixnw suxnothta me to arxeio  file_transformer.py
	4)Ta organwnw se fakelous me to arxeio kwdika sorting_individuals.py '''

def main():
    subprocess.call(["python","file_extractor.py"])
    subprocess.call(["python","file_organizer.py"])
    subprocess.call(["python","file_transformer.py"])
    subprocess.call(["python","file_individual_sorter.py"])
    
    print("Job done nicely :)")

    return 0


if __name__ == '__main__':
    status = main()
    sys.exit(status)
