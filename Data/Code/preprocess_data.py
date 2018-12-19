from __future__ import print_function
from shutil import copy2
from definitions import FILES_DIR
import csv
import librosa
import librosa.display
import os
import numpy as np
import sys
from librosa.feature import mfcc
from librosa.feature import delta

''' This script can produce 3 possible dataset types so far:

        1) Spectro: spectrogram input of size (:,:,128,400)
        2) StackPerRow: Mfcc Delta Delta2 stacked per row input of size (:,:,60,400)
        3) StackPlain: Mfcc Delta Delta2 plain stacked input of size (:,:,60,400)

    Also it creates an input csv with the corresponding name'''


def _get_class_names(path):  # class names are subdirectory names in Samples/ directory
    class_names = os.listdir(path)
    return class_names


def createCsv(data):
    with open("Inputs.csv", "wb") as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow([row])


def moveSamples(list, source, dest):
    for filename in list:
        if not os.path.exists(dest + '/' + filename):
            copy2(source + '/' + filename, dest)
        else:
            print("File already exists.")


def createAudioSamplesDir(nb_files):
    '''Create Audio Sample directory
        Create Csv with shuffled inputs'''

    os.chdir(FILES_DIR)
    audio_sample_path = "Audio_Files_Samples/"
    audio_origin_path = "Audio_Files/"
    inputFiles = []
    if not os.path.exists(audio_sample_path):
        print("There is no samples directory. \nCreating new one..")
        os.mkdir(audio_sample_path)

    class_names = _get_class_names(audio_origin_path)

    print("class_names = ", class_names)

    for idx, classname in enumerate(class_names):  # go through the subdirs

        class_files = os.listdir(audio_origin_path + classname)

        if classname.startswith("H"):
            choice = np.random.choice(class_files, 556,
                                      replace=False)  # randomly select n files from a class
        elif classname.startswith("P"):
            choice = np.random.choice(class_files, 278,
                                      replace=False)
        else:
            sys.exit("Unable to determine audio class : {:14s} ".format(classname))

        if not os.path.exists(audio_sample_path + classname):
            print("There is no {:s} sub-directory. \nCreating new one..".format(classname))
            os.mkdir(audio_sample_path + classname)

        moveSamples(choice, audio_origin_path + classname,
                    audio_sample_path + classname)  # copy from audio origin to audio samples
        inputFiles.extend(choice)

    inputFiles = np.array(inputFiles)
    np.random.shuffle(inputFiles)

    # createCsv(inputFiles)

    return audio_sample_path


def _LogMelSpectrogram(aud, sr):
    melgram = librosa.logamplitude(
        librosa.feature.melspectrogram(
            aud, sr=sr, n_mels=96), ref_power=1.0)  # load log melspectrogram and save it with 128 mels
    # also reashape to [:,:,:,400]
    return melgram




def extract_feature_array(aud, sr):
    mfcc_feat = mfcc(aud, sr, n_mfcc=20)
    delta_feat = delta(mfcc_feat)
    delta2_feat = delta(delta_feat, order=2)
    feature = np.concatenate((mfcc_feat, delta_feat, delta2_feat))
    # feature = np.mean(feature, axis=1)
    return feature


class preprocess_dataset(object):
    def __init__(self, nb_files, path):
        os.chdir(FILES_DIR)
        if os.path.exists(path):
            self.AudioSamplePath = "Audio_Files_Samples/"
        else:
            print("Creating new Audio Sample Files..\n")
            self.AudioSamplePath = createAudioSamplesDir(nb_files)

    def _createDataset(self, printevery, outpath):

        if not os.path.exists(outpath):
            os.mkdir(outpath)  # make a new directory for preproc'd files
        else:
            print("A dataset with extracted features already exists.")
            #    print("Path: {:s} alraedy exists.".format(outpath))
            return outpath

        data = []
        class_names = _get_class_names(path=self.AudioSamplePath)  # get the names of the subdirectories
        nb_classes = len(class_names)
        print("class_names = ", class_names)
        for idx, classname in enumerate(class_names):  # go through the subdirs

            if not os.path.exists(outpath + classname):
                os.mkdir(outpath + classname)  # make a new subdirectory for preproc class

            class_files = os.listdir(self.AudioSamplePath + classname)
            n_files = len(class_files)
            n_load = n_files
            print(' class name = {:14s} - {:3d}'.format(classname, idx),
                  ", ", n_files, " files in this class", sep="")

            for idx2, infilename in enumerate(class_files):  # fore each file
                audio_path = self.AudioSamplePath + classname + '/' + infilename
                if (0 == idx2 % printevery):
                    print('\r Creating class: {:14s} ({:2d} of {:2d} classes)'.format(classname, idx + 1, nb_classes),
                          ", file ", idx2 + 1, " of ", n_load, ": ", audio_path, sep="")
                aud, sr = librosa.load(audio_path)

                if outpath == "Extracted_features/":
                    if len(aud) == 0: # if empty or corrupted file
                        print("File remove!! : ", audio_path)
                        os.remove(audio_path)
                    else:
                        data = extract_feature_array(aud, sr)
                else:
                    sys.exit("Unable to specify outpath")
                outfile = outpath + classname + '/' + infilename + '.npy'

                np.save(outfile, data)  # save output to file

        return outpath


if __name__ == '__main__':
    audio = FILES_DIR + "/" + "Audio_Files" + "/HC/0a21ce6f-a5e2-413c-a1a1-12dce0eb8826_fals.wav"
    aud, sr = librosa.load(audio)
    feature = extract_feature_array(aud, sr)
    # feature = np.matrix(feature)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5.2))
    plt.show()
    print(feature.shape)

    # feature = np.vstack((mfcc, fdelta, fdelta2))
