from __future__ import print_function

from sklearn.model_selection import train_test_split

from definitions import FILES_DIR
import numpy as np
import sys
import os


def get_class_names(path):
    print(path)  # class names are subdirectory names in Preproc directory
    class_names = os.listdir(path)
    return class_names


def get_total_files(path, train_percentage):
    sum_total = 0
    sum_train = 0
    sum_test = 0
    subdirs = os.listdir(path)
    for subdir in subdirs:
        files = os.listdir(path + subdir)
        n_files = len(files)
        sum_total += n_files
        n_train = int(train_percentage * n_files)
        n_test = n_files - n_train
        sum_train += n_train
        sum_test += n_test

    return sum_total, sum_train, sum_test


def get_sample_dimensions(path):
    classname = os.listdir(path)[0]
    files = os.listdir(path + classname)
    infilename = files[0]
    audio_path = path + classname + '/' + infilename
    mel = np.load(audio_path)
    return mel.shape


def encode_class(class_name, class_names):  # makes a "one-hot" vector for each class name called
    try:
        idx = class_names.index(class_name)
        vec = np.zeros(len(class_names))
        vec[idx] = 1
        return vec
    except ValueError:
        return None


def shuffle_XY_paths(X, Y, paths):  # generates a randomized order, keeping X&Y(&paths) together
    assert (X.shape[0] == Y.shape[0])
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    newpaths = paths
    for i in range(len(idx)):
        newX[i] = X[idx[i], :, :]
        newY[i] = Y[idx[i], :]
        newpaths[i] = paths[idx[i]]
    return newX, newY, newpaths

def get_min_dimensions(path):
    class_names = os.listdir(path)  # get the names of the subdirectories
    min = 80000
    for idx, classname in enumerate(class_names):  # go through the subdirs
        class_files = os.listdir(path + classname)
        for idx2, infilename in enumerate(class_files):  # fore each file
            mel_path = path + classname + '/' + infilename
            mel = np.load(mel_path)
            if min > mel.shape[1]:
                min = mel.shape[1]
                print(infilename)
                print("Minimum shape is : ", mel.shape)
    mel = mel[:, 0:min]
    print("Minimum shape is : ", mel.shape)
    return mel.shape

'''To make sure statistics in training & testing are as similar 
as possible I create train and test dataset separately'''


def build_dataset(train_percentage, path):
    os.chdir(FILES_DIR)

    class_names = get_class_names(path=path)
    print("class_names = ", class_names)

    total_files, total_train, total_test = get_total_files(path=path, train_percentage=train_percentage)
    print("total files = ", total_files)

    nb_classes = len(class_names)



    mel_dims = get_min_dimensions(path)  # Find out the 'shape' of smallest data file

    # pre-allocate memory for speed (old method used np.concatenate, slow)
    paths_train = []
    paths_test = []
    X = np.zeros((total_files, mel_dims[0], mel_dims[1]))
    Y = np.zeros((total_files, nb_classes))
    train_count = 0
    test_count = 0
    count = 0
    for idx, classname in enumerate(class_names):
        this_Y = np.array(encode_class(classname, class_names))  # makes one hot vector
        this_Y = this_Y[np.newaxis, :]  # reshape
        print(this_Y)
        class_files = os.listdir(path + classname)
        n_files = len(class_files)
        n_load = n_files
        n_train = int(train_percentage * n_load)
        printevery = 50
        print("")
        for idx2, infilename in enumerate(class_files[0:n_load]):
            mel_path = path + classname + '/' + infilename
            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname, idx + 1, nb_classes),
                      ", file ", idx2 + 1, " of ", n_load, ": ", mel_path, sep="")
            # start = timer()
            mel = np.load(mel_path)
            mel = mel[:, 0:mel_dims[1]]
            # because files may be differnt size: clip to smallest file size and reshape
            # end = timer()
            # print("time = ",end - start)
            X[count, :, :] = mel
            Y[count, :] = this_Y
            count += 1

        print("")
    print(X.shape)
    print("Shuffling order of data...")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=123, train_size=train_percentage)

    # X_train, Y_train, paths_train = shuffle_XY_paths(X_train, Y_train, paths_train)
    # X_test, Y_test, paths_test = shuffle_XY_paths(X_test, Y_test, paths_test)

    return X_train, y_train, paths_train, X_test, y_test, paths_test, class_names
