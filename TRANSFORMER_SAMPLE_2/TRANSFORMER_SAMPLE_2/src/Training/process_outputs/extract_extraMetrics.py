import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


import sklearn.metrics as sklearnMetrics
import math
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
import scipy
#1d array-like
def getPrecision(y_true, y_pred,average='weighted'): #average=None if we want metric per class
    precision = sklearnMetrics.precision_score(y_true, y_pred=y_pred, average=average) #precission per class
    #sklearnMetrics.precision_score(y_true, y_pred=pred, average='weighted') #precission in general
    return precision

def getRecall(y_true, y_pred,average='weighted'):
    recall = sklearnMetrics.recall_score(y_true, y_pred=y_pred, average=average) #precission per class
    #sklearnMetrics.recall_score(y_true, y_pred=pred, average='weighted') #precission in general
    return recall

def getConfussionMatrix(y_true, y_pred, classes):
    cm = sklearnMetrics.confusion_matrix(y_true, y_pred, labels=classes)
    return cm

def getAccucacy(y_true, y_pred):
    accuracy = sklearnMetrics.accuracy_score(y_true, y_pred)
    return accuracy

def getF1(y_true, y_pred, average='weighted'):
    f1 = sklearnMetrics.f1_score(y_true, y_pred, average=average)
    return f1


def get_summary_metrics(train_pred, labels_train, test_pred, labels_test,classes= [1,-1], saveData=False, path2save = ""):
    precission_train = getPrecision(y_pred=train_pred, y_true=labels_train)
    recall_train = getRecall(y_pred=train_pred, y_true=labels_train)
    accuracy_train = getAccucacy(y_pred=train_pred, y_true=labels_train)
    F1_train = getF1(y_pred=train_pred, y_true=labels_train)
    confussion_matrix_train = getConfussionMatrix(y_pred=train_pred, y_true=labels_train, classes=classes)
    precission_test =getPrecision(y_pred=test_pred, y_true=labels_test)
    recall_test = getRecall(y_pred=test_pred, y_true=labels_test)
    accuracy_test = getAccucacy(y_pred=test_pred, y_true=labels_test)
    F1_test = getF1(y_pred=test_pred, y_true=labels_test)
    confussion_matrix_test = getConfussionMatrix(y_pred=test_pred, y_true=labels_test, classes=classes)
    print("....................METRICS.................................")
    print(path2save)
    print("------TRAIN-------")
    print('Precission train: ' + str(precission_train))
    print('Recall train: ' + str(recall_train))
    print("Accuracy train" + str(accuracy_train))
    print("F1 train" + str(F1_train))
    print("Confussion matrix train:" + str(confussion_matrix_train))
    print("-------TEST-------")
    print('Precission test: ' + str(precission_test))
    print('Recall test: ' + str(recall_test))
    print("Accuracy test" + str(accuracy_test))
    print("F1 test" + str(F1_test))
    print("Confussion matrix test:" + str(confussion_matrix_test))
    print("........................................................")
    if(saveData):
        if(path2save==""):
            path2save= "/home/cris/PycharmProjects/InterSpeech19/data/results/baseline/default.txt"
        with open(path2save, "w") as f:
            f.write("-----------TRAIN-------"+"\n")
            f.write("Precission: "+str(precission_train)+"\n")
            f.write("Recall: " + str(recall_train)+"\n")
            f.write("Accuracy:" + str(accuracy_train)+"\n")
            f.write("F1:" + str(F1_train)+"\n")
            f.write("Classes: " + str(classes)+"\n")
            f.write("Confussion matrix train:" + str(confussion_matrix_train)+"\n")
            f.write("-----------TEST-------"+"\n")
            f.write("Precission test: " + str(precission_test)+"\n")
            f.write("Recall test: " + str(recall_test)+"\n")
            f.write("Accuracy test" + str(accuracy_test)+"\n")
            f.write("F1 test" + str(F1_test)+"\n")
            f.write("Classes: " + str(classes) + "\n")
            f.write("Confussion matrix test:" + str(confussion_matrix_test)+"\n")
        with open("/home/cris/PycharmProjects/InterSpeech19/data/results/baseline/common_metrics.txt", "a+") as f:
            f.write(path2save+";"+str(confussion_matrix_train).replace("\n","")+";"+str(confussion_matrix_test).replace("\n","")+"\n")
    return precission_train, recall_train, accuracy_train, F1_train, confussion_matrix_train, precission_test, recall_test, accuracy_test, F1_test, confussion_matrix_test



def get_eval_metrics(predictions,labels):
    avg_accuracy = round(accuracy_score(y_pred=predictions, y_true=labels) * 100, 2)
    weighted_f1 = round(f1_score(y_pred=predictions, y_true=labels, average='weighted') * 100, 2)
    micro_f1 = round(f1_score(y_pred=predictions, y_true=labels, average='micro') * 100, 2)
    macro_f1 = round(f1_score(y_pred=predictions, y_true=labels, average='macro') * 100, 2)
    precision_SC = round(precision_score(y_pred=predictions, y_true=labels, average='macro') * 100, 2)
    recall_SC = round(recall_score(y_pred=predictions, y_true=labels, average='macro') * 100, 2)
    print("Average acc:{}, weighted F1 {}, micro F1 {}, macro F1 {}, Precision {}, Recall {},"
          .format(avg_accuracy, weighted_f1, micro_f1, macro_f1, precision_SC, recall_SC))
    return avg_accuracy, weighted_f1, micro_f1, macro_f1, precision_SC, recall_SC



def calculate_CI(metric_score, n_samples, confidence = 0.95):
      # Change to your desired confidence level
    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    CI = z_value * np.sqrt((metric_score * (100 - metric_score)) / n_samples)
    print("CI: ", str(CI))
    return CI



if __name__ == "__main__":
    glosses_count = "/run/user/1000/gvfs/smb-share:server=137.250.171.20,share=datasets/AVASAG_dictionary/videos_glosses_from_sentences_extra/datasets_files/countGlosses.csv"
    df_glosses_count = pd.read_csv(glosses_count, sep=";", header=0)


    #path_glosses_summary = "/src/Training/out-checkpoints/AVASAG100_originalSpoterPE_land-75/posteriors/checkpoint_t_8/val/summary_preds.csv"
    dataset_name = "WLASL100"
    model2use = "regularTransformerNoPE"
    PE = False
    complete_block = False
    posteriors_folder = dataset_name+"_" + model2use + "_land-75_v1_cblock-" + str(int(complete_block)) + "_PE-" + str(int(PE)) + "_bs-1"  #"AVASAG_" + model2use + "_land-75_v1_cblock-" + str(int(complete_block)) + "_PE-" + str(int(PE)) + "_bs-1" #"AVASAG_regularTransformer_land-75_v1" #
    checkpoint_folder = "checkpoint_v_8"
    split2test = "test"
    path_glosses_summary = "/home/cristinalunaj/PycharmProjects/AVASAG_processing/src/Training/out-checkpoints/" + posteriors_folder + "/posteriors/" + checkpoint_folder + "/" + split2test + "/summary_preds.csv"  # "/src/Training/out-checkpoints/AVASAG100_originalSpoterPE_land-75/posteriors/checkpoint_t_8/val/summary_preds.csv"




    df_glosses_summary = pd.read_csv(path_glosses_summary, sep=",", header=0)

    dictionary_number_names = dict(zip(df_glosses_count.index, df_glosses_count["glosse"]))
    dictionary_names_numbers = dict(zip(df_glosses_count["glosse"], df_glosses_count.index))

    preds = df_glosses_summary["pred"]
    labels = df_glosses_summary["label"]
    print("samples:", len(preds))

    get_summary_metrics(preds, labels, preds, labels, classes=list(range(0,100)), saveData=False, path2save="")
    get_eval_metrics(preds, labels)
    # val_samples = 890 #849
    # calculate_CI(71.35, val_samples, confidence=0.95)












