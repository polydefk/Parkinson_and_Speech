import os

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from Data import preprocess_dataset, build_dataset
from definitions import ROOT_DIR
import numpy as np
from Build_Model import build_model as model
from scipy import interp
from itertools import product


def plot_confusion_matrix(test_model, normalize=False, cmap=plt.cm.Blues, majority=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    def pred_Majority_Voting(predictions):
        num_outputs = len(predictions['resNET'])
        pred = np.zeros(num_outputs, dtype=np.int32)

        for i in range(num_outputs):
            if (predictions['resNET'][i] == 1 and predictions["CRNN"][i] == 1):
                pred[i] = 1
            # elif (predictions['deepLSTM'][i] == 1 and predictions["CRNN"][i] == 1):  # only with deepLSTM
            #     pred[i] = 1
            # elif (predictions['deepLSTM'][i] == 1 and predictions["resNET"][i] == 1):
            #     pred[i] = 1
            else:
                pred[i] = 0

        return pred

    def get_confusion_matrix_one_hot():
        '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
        where truth is 0/1, and max along each row of model_results is model result
        '''
        if majority == True:
            pred = dict()
            models = ["resNET", "CRNN"]
            os.chdir(ROOT_DIR)
            for modelaki in models:
                pred[modelaki] = np.load('prediction_test_' + modelaki + '.npy')
            predictions = pred_Majority_Voting(pred)
        else:
            predictions = np.argmax(test_model.pred, axis=1)

        num_outputs = test_model.Y_test.shape[1]
        confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)

        assert len(predictions) == test_model.Y_test.shape[0]

        for actual_class in range(num_outputs):
            idx_examples_this_class = test_model.Y_test[:, actual_class] == 1
            prediction_for_this_class = predictions[idx_examples_this_class]
            for predicted_class in range(num_outputs):
                count = np.sum(prediction_for_this_class == predicted_class)
                confusion_matrix[actual_class, predicted_class] = count
        assert np.sum(confusion_matrix) == len(test_model.Y_test)
        assert np.sum(confusion_matrix) == np.sum(test_model.Y_test)
        return confusion_matrix

    plt.figure(figsize=(7, 5.2))
    cm = get_confusion_matrix_one_hot()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.colorbar()
    tick_marks = np.arange(len(test_model.nb_classes))
    plt.xticks(tick_marks, test_model.nb_classes, rotation=45)
    plt.yticks(tick_marks, test_model.nb_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if majority:
        # plt.savefig(ROOT_DIR + '/' + 'Majority' + '_Confusion.png')
        plt.title('Confusion Matrix of after majority voting')
        pass
    else:
        # plt.savefig(ROOT_DIR + '/' + test_model.model_name + '_Confusion.png')
        plt.title('Confusion Matrix of ' + test_model.model_name)
        pass


def plot_ROC(test_model):
    plt.figure()
    fpr = dict()
    tpr = dict()
    lw = 2
    roc_auc = dict()
    for i in range(len(test_model.nb_classes)):
        fpr[i], tpr[i], _ = roc_curve(test_model.Y_test[:, i], test_model.pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_model.Y_test.ravel(), test_model.pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(test_model.nb_classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(test_model.nb_classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(test_model.nb_classes)

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
    plt.title('Roc Curve for ' + test_model.model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # plt.savefig(DISS_CODE_FIG_DIR + '/' + model_name + '_ROC.png')


if __name__ == '__main__':
    models = ["resNET", "CRNN"]  # "resNET", "CRNN", "deepLSTM", "CRNN"
    predictions = dict()

    for modelaki in models:
        np.random.seed(12345)
        nb_files = 278 + 556
        batch_size = 32
        nb_epoch = 10
        train_percentage = 0.8
        model_name = modelaki
        monitor = 'accuracy'

        dataset = preprocess_dataset(nb_files=nb_files, path="Audio_Files_Samples/")

        # get the data
        X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names = \
            build_dataset(train_percentage=train_percentage,
                          path=dataset._createDataset(printevery=50,
                                                      outpath="Extracted_features/"))


        print(class_names)
        # make the model
        test_model = model(class_names, X_train, Y_train, X_test, Y_test, nb_epoch=nb_epoch, batch_size=batch_size,
                           model_name=model_name, monitor=monitor)
        # load weights
        test_model.model_type.load_weights(test_model.model_name + '_weights.hdf5')

        # make predictions
        test_model.pred = test_model.model_type.predict(test_model.X_test, verbose=0)

        # encode predictiosn
        pred = np.argmax(test_model.pred, 1)

        # save to dictionary
        predictions[modelaki] = pred
        os.chdir(ROOT_DIR)
        np.save('prediction_test_' + modelaki, predictions[modelaki])

        # plot roc
        plot_ROC(test_model)

        # plot confusion matrix
        plot_confusion_matrix(test_model, majority=False)

    # plot confusion matrix with majority
    plot_confusion_matrix(test_model, majority=True)
    plt.show()
