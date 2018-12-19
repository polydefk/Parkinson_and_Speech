from __future__ import print_function

import csv

import numpy as np
import shutil

from definitions import MODEL_RUN_DIR, DISS_CODE_FIG_DIR

import matplotlib.pyplot as plt
from keras.utils import plot_model
from Data.Code import build_dataset, preprocess_dataset
from Build_Model import build_model as model


def main():
    models = ["CRNN", "deepLSTM", "deepCNN", "resNET", "BiLSTM"]  # "CRNN","deepLSTM", "deepCNN", "resNET","BiLSTM",
    header = ["model", 'fscore', 'roc', 'accuracy', 'precision', 'recall', 'epochs']

    for modelaki in models:
        shutil.copy(__file__, MODEL_RUN_DIR)

        np.random.seed(1234)
        nb_files = 278 + 556
        batch_size = 32
        nb_epoch = 25
        train_percentage = 0.8
        model_name = modelaki
        monitor = 'accuracy'  # deepLSTM , deepCNN  , resNET , CRNN, BLSTM

        dataset = preprocess_dataset(nb_files=nb_files, path="Audio_Files_Samples/")

        # get the data
        X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names = \
            build_dataset(train_percentage=train_percentage,
                          path=dataset._createDataset(printevery=10,
                                                      outpath="Extracted_features/"))

        # make the model
        test_model = model(class_names, X_train, Y_train, X_test, Y_test, nb_epoch=nb_epoch, batch_size=batch_size,
                           model_name=model_name, monitor=monitor)

        print("-------------------------> Training for model : ", test_model.model_name, " <-------------------------")

        # train the model
        history = test_model.train(load_checkpoint=True, ifearlystop=True)

        # Plot the model
        # plot_model(test_model.model_type, to_file=DISS_CODE_FIG_DIR + '/' + model_name + '.png')

        # Evaluate the model
        test_model.evaluate(history)

        # plot data with PCA
        # test_model.plot_PCA()

        # Plot the average receiver operating characteristic for each class
        # test_model.plot_ROC()

        np.set_printoptions(precision=2)  # set the precision of floats in 2 decimeter
        ## Plot non-normalized confusion matrix

        # test_model.plot_confusion_matrix(normalize=False)

    #plt.tight_layout()
    #plt.show()
    plt.close()


if __name__ == '__main__':
    main()
