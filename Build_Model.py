from __future__ import print_function
import keras
import shutil
import csv
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *
from keras.models import *
from keras.engine import Input
from keras.layers import Conv2D, Flatten, AveragePooling2D
from keras.layers.advanced_activations import ELU
from  keras.losses import categorical_crossentropy
from keras.layers import MaxPooling2D
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from os.path import isfile
from sklearn.metrics import roc_curve, auc
from itertools import cycle, product
from keras.layers.core import Dense, Activation, Dropout, Reshape
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from sklearn import tree
from definitions import MODEL_RUN_DIR, DISS_CODE_FIG_DIR
import time

from matplotlib.colors import ListedColormap


class build_model(object):
    def __init__(self, nb_classes, X_train, Y_train, X_test, Y_test, nb_epoch, batch_size=64, model_name='',
                 monitor=''):
        shutil.copy(__file__, MODEL_RUN_DIR)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.monitor = monitor
        self.model_name = model_name
        if model_name == "deepLSTM":
            self.model_type = self.deep_lstm()
        elif model_name == "deepCNN":
            self.model_type = self.deep_cnn()
        elif model_name == "resNET":
            self.model_type = self.resNET()
        elif model_name == "CRNN":
            self.model_type = self.crnn()
        elif model_name == "exTraTree":
            self.model_type = self.exTraTree()
        elif model_name == "BiLSTM":
            self.model_type = self.BiLSTM()
        else:
            print("No proper model is defined")
            return

    def deep_lstm(self):
        xShape = self.X_train.shape[1]
        yShape = self.X_train.shape[2]
        model = Sequential()
        model.add(LSTM(512, input_shape=(xShape, yShape), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[self.monitor])
        model.summary()
        return model

    def deep_cnn(self):
        nb_filters = 60  # number of convolutional filters to use
        pool_size = (2, 2)  # size of pooling area for max pooling
        kernel_size = (3, 3)  # convolution kernel size
        nb_layers = 5
        input_shape = (self.X_train.shape[1], self.X_train.shape[2], 1)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)

        model = Sequential()
        conv = Conv2D(nb_filters, input_shape=input_shape, kernel_size=kernel_size)
        model.add(conv)
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))

        for layer in range(nb_layers - 1):
            model.add(Conv2D(nb_filters, kernel_size=kernel_size))
            model.add(BatchNormalization(axis=1))
            model.add(ELU(alpha=1.0))
            model.add(MaxPooling2D(pool_size=pool_size))
            model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.nb_classes)))
        model.add(Activation("softmax"))

        model.summary()
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=0.0004),
                      metrics=[self.monitor])

        return model

    def crnn(self):
        input_shape = (self.X_train.shape[1], self.X_train.shape[2], 1)
        cnn_layer_num_filters = 180
        cnn_layer_kernel_size = (3, 3)
        cnn_layer_strides = (4, 4)
        rnn_cell = Bidirectional(LSTM(25, return_sequences=True))
        full_connected_layer_num_units = 120
        dropout = 0.2
        opt = Adam
        learning_rate = 0.0005

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)

        model = Sequential()
        model.add(Conv2D(filters=cnn_layer_num_filters, kernel_size=cnn_layer_kernel_size,
                         strides=cnn_layer_strides, padding="valid", input_shape=input_shape))
        print(model.output_shape)
        model.add(BatchNormalization(axis=2))
        model.add(Activation("relu"))
        model.summary()
        print(model.output_shape)
        model.add(Reshape((963 * 2 * 3, 50)))

        model.add(rnn_cell)

        model.add(Bidirectional(LSTM(60, return_sequences=False)))

        model.add(Dense(units=full_connected_layer_num_units))
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(units=2, activation="softmax"))

        optimizer = opt(lr=learning_rate)

        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[self.monitor])
        model.summary()
        return model

    def exTraTree(self):
        from sklearn.ensemble import ExtraTreesRegressor
        pca = PCA(n_components=60)
        estimator = ExtraTreesRegressor(n_estimators=834, max_features=60)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1] * self.X_train.shape[2])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2])
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.fit_transform(self.X_test)
        return estimator

    def BiLSTM(self):

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[2], self.X_train.shape[1])
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2], self.X_test.shape[1])
        xShape = self.X_train.shape[1]
        yShape = self.X_train.shape[2]

        model = Sequential()
        model.add(Bidirectional(LSTM(120, return_sequences=True), input_shape=(xShape, yShape)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(60, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(30, return_sequences=True)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(15, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

        # try using different optimizers and different optimizer configs
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[self.monitor])
        model.summary()
        return model

    def resNET(self):

        def resnet_layer(inputs,
                         num_filters=16,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         batch_normalization=True,
                         conv_first=True):
            """2D Convolution-Batch Normalization-Activation stack builder
            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization
                conv_first (bool): conv-bn-activation (True) or
                    bn-activation-conv (False)
            # Returns
                x (tensor): tensor as input to the next layer
            """
            conv = Conv2D(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))
            x = inputs
            if conv_first:
                x = conv(x)
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
            else:
                if batch_normalization:
                    x = BatchNormalization()(x)
                if activation is not None:
                    x = Activation(activation)(x)
                x = conv(x)
            return x

        def resnet_v1(input_shape, depth, num_classes=2):

            if (depth - 2) % 6 != 0:
                raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
            # Start model definition.
            num_filters = 16
            num_res_blocks = int((depth - 2) / 6)

            inputs = Input(shape=input_shape)
            x = resnet_layer(inputs=inputs)
            # Instantiate the stack of residual units
            for stack in range(3):
                for res_block in range(num_res_blocks):
                    strides = 1
                    if stack > 0 and res_block == 0:  # first layer but not first stack
                        strides = 2  # downsample
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     strides=strides)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)
                    if stack > 0 and res_block == 0:  # first layer but not first stack
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = resnet_layer(inputs=x,
                                         num_filters=num_filters,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=None,
                                         batch_normalization=False)
                    x = keras.layers.add([x, y])
                    x = Activation('relu')(x)
                num_filters *= 2

            # Add classifier on top.
            # v1 does not use BN after last shortcut connection-ReLU
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            return model

        def resnet_v2(input_shape, depth=29, num_classes=2):

            if (depth - 2) % 9 != 0:
                raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
            # Start model definition.
            num_filters_in = 16
            num_res_blocks = int((depth - 2) / 9)

            inputs = Input(shape=input_shape)
            # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
            x = resnet_layer(inputs=inputs,
                             num_filters=num_filters_in,
                             conv_first=True)

            # Instantiate the stack of residual units
            for stage in range(3):
                for res_block in range(num_res_blocks):
                    activation = 'relu'
                    batch_normalization = True
                    strides = 1
                    if stage == 0:
                        num_filters_out = num_filters_in * 4
                        if res_block == 0:  # first layer and first stage
                            activation = None
                            batch_normalization = False
                    else:
                        num_filters_out = num_filters_in * 2
                        if res_block == 0:  # first layer but not first stage
                            strides = 2  # downsample

                    # bottleneck residual unit
                    y = resnet_layer(inputs=x,
                                     num_filters=num_filters_in,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=activation,
                                     batch_normalization=batch_normalization,
                                     conv_first=False)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters_in,
                                     conv_first=False)
                    y = resnet_layer(inputs=y,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     conv_first=False)
                    if res_block == 0:
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = resnet_layer(inputs=x,
                                         num_filters=num_filters_out,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=None,
                                         batch_normalization=False)
                    x = keras.layers.add([x, y])

                num_filters_in = num_filters_out

            # Add classifier on top.
            # v2 has BN-ReLU before Pooling
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            return model

        input_shape = (self.X_train.shape[1], self.X_train.shape[2], 1)

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1)
        model = resnet_v1(input_shape=input_shape, depth=20, num_classes=2)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001), metrics=[self.monitor])
        model.summary()
        return model

    # pca
    def plot_PCA(self):
        def plot_hyperplane(clf, min_x, max_x, linestyle, label):

            # get the separating hyperplane
            w = clf.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
            yy = a * xx - (clf.intercept_[0]) / w[1]
            plt.plot(xx, yy, linestyle, label=label)

        def plot_figure(X, Y, title, transform):
            if transform == "pca":
                X = PCA(n_components=2).fit_transform(X)
            elif transform == "cca":
                X = CCA(n_components=2).fit(X, Y).transform(X)
            else:
                raise ValueError

            min_x = np.min(X[:, 0])
            max_x = np.max(X[:, 0])

            min_y = np.min(X[:, 1])
            max_y = np.max(X[:, 1])

            classif = OneVsRestClassifier(SVC(kernel='linear'))
            classif.fit(X, Y)

            plt.title(title)

            zero_class = np.where(Y[:, 0])
            one_class = np.where(Y[:, 1])
            plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
            plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                        facecolors='none', linewidths=2, label='HC')
            plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                        facecolors='none', linewidths=2, label='PD')

            plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                            'Boundary\nfor HC')
            plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                            'Boundary\nfor PD')
            plt.xticks(())
            plt.yticks(())

            plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
            plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)

            plt.xlabel('First principal component')
            plt.ylabel('Second principal component')
            plt.legend(loc="upper left")
            #plt.savefig(DISS_CODE_FIG_DIR + '/' + 'pca.png')

        plt.figure()
        print("Plotting PCA")
        print("=" * 65)

        X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2])
        plot_figure(X_test, self.Y_test, "PCA", "pca")

    # ROC
    def plot_ROC(self):
        plt.figure()
        fpr = dict()
        tpr = dict()
        lw = 2
        roc_auc = dict()
        for i in range(len(self.nb_classes)):
            fpr[i], tpr[i], _ = roc_curve(self.Y_test[:, i], self.pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.Y_test.ravel(), self.pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(self.nb_classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(self.nb_classes)):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(self.nb_classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        color = 'darkorange'
        plt.plot(fpr["micro"], tpr["micro"],
                 label='ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color=color, linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Roc Curve for ' + self.model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        #plt.savefig(DISS_CODE_FIG_DIR + '/' + self.model_name + '_ROC.png')

     # Confusion Matric

    def plot_confusion_matrix(self, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        def get_confusion_matrix_one_hot():
            '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
            where truth is 0/1, and max along each row of model_results is model result
            '''
            assert self.pred.shape == self.Y_test.shape
            num_outputs = self.Y_test.shape[1]
            confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
            predictions = np.argmax(self.pred, axis=1)
            assert len(predictions) == self.Y_test.shape[0]

            for actual_class in range(num_outputs):
                idx_examples_this_class = self.Y_test[:, actual_class] == 1
                prediction_for_this_class = predictions[idx_examples_this_class]
                for predicted_class in range(num_outputs):
                    count = np.sum(prediction_for_this_class == predicted_class)
                    confusion_matrix[actual_class, predicted_class] = count
            assert np.sum(confusion_matrix) == len(self.Y_test)
            assert np.sum(confusion_matrix) == np.sum(self.Y_test)
            return confusion_matrix

        plt.figure(figsize=(7, 5.2))
        cm = get_confusion_matrix_one_hot()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        #np.save(MODEL_RUN_DIR + '_' + self.model_name + '_ConfusionMatrixNumpy', cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)

        plt.colorbar()
        tick_marks = np.arange(len(self.nb_classes))
        plt.xticks(tick_marks, self.nb_classes, rotation=45)
        plt.yticks(tick_marks, self.nb_classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.title('Confusion Matrix of ' + self.model_name)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.savefig(DISS_CODE_FIG_DIR + '/' + self.model_name + '_Confusion.png')

    # fscore accuracy loss, precision, recall, support
    def evaluate(self, history):
        print(history.history.keys())

        # summarize history for accuracy
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy '+self.model_name)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.savefig(DISS_CODE_FIG_DIR + '/' + self.model_name + '_Accuracy.png')

        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss '+self.model_name)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.savefig(DISS_CODE_FIG_DIR + '/' + self.model_name + '_Loss.png')

        self.pred = self.model_type.predict(self.X_test, verbose=0)

        predict = np.argmax(self.pred, 1)
        y_true = np.argmax(self.Y_test, 1)

        # evaluate the model
        _, accuracy = self.model_type.evaluate(self.X_test, self.Y_test, batch_size=32)
        print("\nAccuracy = {:.2f}".format(accuracy))

        # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
        p, r, f, s = precision_recall_fscore_support(y_true, predict, average='micro')
        print("Outputs from Sklearn Lib")
        print("F-Score: ", round(f, 3))
        print("Precision: ", round(p, 3))
        print("Recall: ", round(r, 3))
        header = ["model", 'fscore', 'roc', 'accuracy', 'precision', 'recall', 'epochs']
        data = [self.model_name, round(f, 3),
                round(accuracy, 3), round(p, 3), round(r, 3), self.nb_epoch]
        with open("outputs.csv", "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerow(data)

    def train(self, load_checkpoint=False, ifearlystop=False):
        # Initialize weights using checkpoint if it exists. (Checkpointing requires h5py)

        print("X train shape = ", self.X_train.shape)
        print("X test shape = ", self.X_test.shape)

        checkpoint_filepath = self.model_name + '_weights.hdf5'
        checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, monitor="val_acc", verbose=1,
                                       save_best_only=True)
        earlystop = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode='auto')

        if (load_checkpoint):
            print("Looking for previous weights...")
            if (isfile(checkpoint_filepath)):
                print('Checkpoint file detected. Loading weights.')
                self.model_type.load_weights(checkpoint_filepath)
            else:
                print('No checkpoint file detected.  Starting from scratch.')
        else:
            print('Starting from scratch (no checkpoint)')

        if (ifearlystop):
            callbacks = [checkpointer, earlystop]
            print('Training with earlystop')
        else:
            callbacks = [checkpointer]
            print('Training without earlystop')

        startTime = time.time()
        if (self.model_name == 'exTraTree'):
            estimator = self.model_type
            estimator.fit(self.X_train, self.Y_train)
            y_pred = estimator.predict(self.X_test)
            from sklearn.metrics import mean_squared_error
            score = mean_squared_error(self.Y_test, y_pred)
            print(round(score, 6))
        else:
            # train and score the model
            fit = self.model_type.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                                      shuffle=True,

                                      validation_data=[self.X_test, self.Y_test], callbacks=callbacks)

        endTime = time.time()
        totalTime = endTime - startTime
        print("-" * 65)
        print("The total time for Training is : ", round(totalTime, 4))
        print("-" * 65)
        print("The average time for each epoch is : ", round(totalTime / self.nb_epoch, 4))
        print("-" * 65)
        return fit


if __name__ == '__main__':
    pass
