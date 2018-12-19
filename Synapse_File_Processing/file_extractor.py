'''Ta arxeia einai sthn arxikh toys morfh opws katevhkan apo to downloader.py '''
import os
import sys
rootDir = '.'
directory = '/home/polaras/Desktop/edw' #<----- to arxiko directory poy 8a tre3ei to script
os.chdir(directory)

def main():
    counter = 0
    for folderName, subfolders, filenames in os.walk(rootDir):
        print('The current folder is ' + folderName)
        for subfolder in subfolders:
            print(subfolder)
        for filename in filenames:
            print('FILE INSIDE ' + folderName + ': ' + filename)
            if filename.endswith('.tmp'): #gia na 3exwrisw apola ta upoloipa arxeia
                counter = counter + 1
                name = subfolder + filename[71:] #71 #o teleutaios subfolder einai kai to id toy hxhtikou kai 8elw na to valw brosta toy
                folderName = folderName[2:] #diwxnw to ./ pou einai to rootDir kai ftiaxnw to old_path kai new_path gia na ta metaonomasw kai metakinisw parallhla
                old_path = os.path.join(directory, folderName)
                old_path = os.path.join(old_path, filename)
                new_path = os.path.join(directory, name)
                print(folderName)
                print(old_path)
                print(new_path)
                os.rename(old_path,new_path) #<------ edw ginetai h doyleia
        print('')
    print(counter)
    return 0

if __name__ == '__main__':
    status = main()
    sys.exit(status)