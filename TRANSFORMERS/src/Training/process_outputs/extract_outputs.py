import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


if __name__ == "__main__":

    dataset_name = "WLASL100"
    model2use = "regularTransformerNoPE"
    PE = False
    complete_block = False
    posteriors_folder = dataset_name + "_" + model2use + "_land-75_v1_cblock-" + str(
        int(complete_block)) + "_PE-" + str(
        int(PE)) + "_bs-1"  # "AVASAG_" + model2use + "_land-75_v1_cblock-" + str(int(complete_block)) + "_PE-" + str(int(PE)) + "_bs-1" #"AVASAG_regularTransformer_land-75_v1" #
    checkpoint_folder = "checkpoint_v_8"
    split2test = "test"

    if (dataset_name == "AVASAG"):
        glosses_count = "/run/user/1000/gvfs/smb-share:server=137.250.171.20,share=datasets/AVASAG_dictionary/videos_glosses_from_sentences_extra/datasets_files/countGlosses.csv"
        df_glosses_count = pd.read_csv(glosses_count, sep=";", header=0)


    elif (dataset_name == "WLASL100"):
        glosses_count = "/home/cristinalunaj/PycharmProjects/AVASAG_processing/data/WLASLS100/WLASL100_v0.3.csv"
        df_glosses_count_aux = pd.read_csv(glosses_count, sep=";", header=0)
        df_glosses_count_aux.rename(columns={"gloss_name": "glosse"},inplace=True)
        df_glosses_count = df_glosses_count_aux[["gloss_number", "glosse"]].drop_duplicates()
        df_glosses_count.reset_index(drop=True, inplace=True)


    dictionary_number_names = dict(zip(df_glosses_count.index, df_glosses_count["glosse"]))
    dictionary_names_numbers = dict(zip(df_glosses_count["glosse"], df_glosses_count.index))

    path_glosses_summary = "/home/cristinalunaj/PycharmProjects/AVASAG_processing/src/Training/out-checkpoints/"+posteriors_folder+"/posteriors/"+checkpoint_folder+"/"+split2test+"/summary_preds.csv" #"/src/Training/out-checkpoints/AVASAG100_originalSpoterPE_land-75/posteriors/checkpoint_t_8/val/summary_preds.csv"
    df_glosses_summary = pd.read_csv(path_glosses_summary, sep=",", header=0)
    # create confusion matrix:
    cm = confusion_matrix(df_glosses_summary["label"], df_glosses_summary["pred"], labels=df_glosses_count.index[0:100])
    # fig = plt.Figure([15,12])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                               display_labels=df_glosses_count.index)
    # disp.plot()
    # plt.show()

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot the confusion matrix using seaborn
    sns.set(rc={'figure.figsize': (45, 40)})  # Size in inches
    sns.heatmap(cm_norm, annot=True, cmap='Blues', fmt='.1f',cbar=False,
                xticklabels=df_glosses_count.glosse[0:100], yticklabels=df_glosses_count.glosse[0:100])

    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix of GAP-Transformer without PE') # Pre-Decoder Query-Transformer without PE
    plt.show()
