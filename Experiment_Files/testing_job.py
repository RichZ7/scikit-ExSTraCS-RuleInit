from h2o.estimators import H2ORandomForestEstimator
from h2o.model.metrics_base import H2OOrdinalModelMetrics
import clutter as RFI
import h2o
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
# h2o.init()
import time
import sklearn
import os
from rfi_ExSTraCS import ExSTraCS, Classifier
from sklearn import metrics
from rfi_ExSTraCS import StringEnumerator as SE
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
import copy


def runRF(data, continuous_attr, nTrees):
    #set up Random forest

    gametes_df = h2o.import_file(data)
    if (continuous_attr==False):
        gametes_df = gametes_df.asfactor()
    gametes_df['Class'] = gametes_df['Class'].asfactor()

    predictors = gametes_df.col_names[:-1]
    response = 'Class'

    start_time = time.time()

    gametes_drf = H2ORandomForestEstimator(
        ntrees=nTrees,
        max_depth=3,
        balance_classes=False,
        min_rows=5,
        seed=1453)

    train, test = gametes_df.split_frame(ratios=[0.8], seed=1453)

    gametes_drf.train(x=predictors,
                      y=response,
                      training_frame=train,
                      validation_frame=test)

    compTime = time.time() - start_time

    # print(gametes_drf.model_performance(test_data=test))

    F1 = gametes_drf.F1(thresholds = [0.5])
    # print(len(F1))
    Precision = gametes_drf.precision(thresholds = [0.5])
    # print(Precision)
    # print(len(Precision))
    Recall = list(gametes_drf.recall(thresholds = [0.5]))
    # print(len(Precision))
    # print(len(Recall))
    Specificity = list(gametes_drf.specificity(thresholds = [0.5]))
    Rec = list(Recall)
    Specif = list(Specificity)
    Rec1 = Rec.pop().pop()
    Specif1 = Specif.pop().pop()
    ROC_AUC = gametes_drf.auc()
    b_accuracy = (Recall.pop().pop() + Specificity.pop().pop())/2

    varimp = gametes_drf.varimp(use_pandas=True)
    varimp.to_csv('Results/' + str(data).split('/')[1] + '/' + str(data).split('/')[2].replace('.txt', '') + '/Visualizations/varimp' + str(nTrees)+ '.csv')
    plotBar('Results/' + str(data).split('/')[1] + '/' + str(data).split('/')[2].replace('.txt', '') + '/Visualizations/varimp' + str(nTrees)+ '.csv')


    return [data, nTrees, compTime, b_accuracy, F1.pop().pop(), Precision.pop().pop(), Rec1, Specif1, ROC_AUC]

def runLCS(data, learning_iter):
    ds = pd.read_csv(data, delimiter = '\t')
    # print(ds.head(10))
    # x = ds.iloc[:,:-1]
    # y = ds.iloc[:, -1]

    # train = ds.iloc[:int(ds.shape[0]*0.8),:]
    # test = ds.iloc[int(ds.shape[0]*0.8):,:]
    train = ds.sample(frac = 0.8)
    test = ds.drop(train.index)
    # print(train.head(10))

    X = train.drop('Class', axis=1).values
    y = train['Class'].values
    X_test = test.drop('Class', axis=1).values
    y_test = test['Class'].values
    dataHeaders = train.drop('Class', axis=1).columns.values

    start_time = time.time()

    model = ExSTraCS(learning_iterations=learning_iter, nu=10, N=2000)

    model.fit(X, y)

    compTime = time.time() - start_time

    # model.export_final_rule_population(ds.drop('Class',axis=1).columns.values, 'Class', filename="fileRulePopulation.csv",DCAL=False)

    visualize(model, data, ds, train, 'lcs')

    pred = model.predict(X_test)

    b_accuracy = metrics.balanced_accuracy_score(y_test, pred)
    F1 = metrics.f1_score(y_test, pred)
    Precision = metrics.precision_score(y_test, pred)
    Recall = metrics.recall_score(y_test, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
    Specificity = tn / (tn + fp)
    ROC_AUC = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return [data, learning_iter, compTime, b_accuracy, F1, Precision, Recall, Specificity, ROC_AUC]

def runRFLCS(data, LCS_param, continuous_attr, ntrees, nIterations):

    compTime, train, test, rCompTime = RFI.RFI_LCS(data, 'ruleTemp.csv', pickle_file='modelTemp', continuous_attr =continuous_attr, classLabel ='Class', ntrees_RFparam=ntrees, max_depth_RFparam=3, min_rows_RFparam=None, balance_classes_RFparam=False, seed_RFparam=1453, number_of_iterations=0)
    if (LCS_param == False):
        lcs = ExSTraCS(learning_iterations = 0,nu=10,N=2000,track_accuracy_while_fit=True, reboot_filename='modelTemp', useRF_Init=True)
    elif (LCS_param == True):
        lcs = ExSTraCS(learning_iterations = nIterations,nu=10,N=2000,reboot_filename='modelTemp', useRF_Init=True)
    # os.remove('ruleTemp.csv')
    train.to_csv('trainTemp.csv', sep=',', index=None)
    test.to_csv('testTemp.csv', sep = ',', index = None)

    # Deprecated
    # train.reset_index(drop=True)
    # test.reset_index(drop=True)
    # X = train.iloc[0:train.shape[1]-2]
    # y = train.iloc[:,train.shape[1]-1]
    # X_test = test.iloc[0:test.shape[1]-2]
    # y_test = test.iloc[:,train.shape[1]-1]
    # print(y)
    # X = X.values
    # y = y.values
    # X_test = X_test.values
    # y_test = y_test.values
    #
    # print(X)
    # print(len(X))
    # print(y)
    # print(len(y))

    converter = SE("trainTemp.csv", "Class")
    headers, classLabel, dataFeatures, dataPhenotypes = converter.get_params()
    RFLCS_nr = lcs.fit(X = dataFeatures,y = dataPhenotypes)
    # print(str(RFLCS_nr.get_final_attribute_tracking_sums())+'\n'+str(ntrees))

    visualize(RFLCS_nr, data, train, train, 'rfilcs')

    instanceLabels = []
    for i in range(dataFeatures.shape[0]):
        instanceLabels.append(i)
    # print(RFLCS_nr.get_attribute_tracking_scores(np.array(instanceLabels)))

    converterT = SE("testTemp.csv", "Class")
    headersT, classLabelT, dataFeaturesT, dataPhenotypesT = converterT.get_params()

    os.remove('modelTemp')

    pred = RFLCS_nr.predict(dataFeaturesT)

    b_accuracy = metrics.balanced_accuracy_score(dataPhenotypesT, pred)
    F1 = metrics.f1_score(dataPhenotypesT, pred)
    Precision = metrics.precision_score(dataPhenotypesT, pred)
    Recall = metrics.recall_score(dataPhenotypesT, pred)
    tn, fp, fn, tp = metrics.confusion_matrix(dataPhenotypesT, pred).ravel()
    Specificity = tn / (tn + fp)
    ROC_AUC = metrics.roc_auc_score(dataPhenotypesT, RFLCS_nr.predict_proba(dataFeaturesT)[:, 1])
    #vis
#     os.remove('iterationData.csv')

    return [data, ntrees, nIterations, compTime, b_accuracy, F1, Precision, Recall, Specificity, ROC_AUC]

def h2oFrame_to_pandas(data):
    data_as_df = h2o.as_list(data, use_pandas=True)
    return data_as_df

def plotBar(path):
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    varimp = pd.read_csv(path)
    labels = list(varimp.iloc[:,1])
    rel_imp = list(varimp.iloc[:,2])
    scaled_imp = list(varimp.iloc[:,3])
    percentage = list(varimp.iloc[:,4])

    for i in range(len(rel_imp)):
        rel_imp[i] = round(rel_imp[i], 2)
        scaled_imp[i] = round(scaled_imp[i], 2)
        percentage[i] = round(percentage[i], 2)

    x = (range(len(labels)))
    n_x = [2 * i for i in x]

    # x = np.arange(len(labels))
    # print(x)
    # # for i in x:
    # #     i += 1
    # x += 2
    # print(x)
    width = .8

    rect = ax.bar(n_x, rel_imp, width, label='Relative Imp')
    rect1 = ax1.bar(n_x, scaled_imp, width, label='Scaled Imp')
    rect2 = ax2.bar(n_x, percentage, width, label='Percentage')

    fig.suptitle('Relative Importance Scores')
    ax.set_xlabel('Feature Name')

    fig1.suptitle('Scaled Importance Scores')
    ax1.set_xlabel('Feature Name')

    fig2.suptitle('Percent Importance Scores')
    ax2.set_xlabel('Feature Name')

    ax.set_xticks(n_x, labels)
    ax1.set_xticks(n_x, labels)
    ax2.set_xticks(n_x, labels)

    plt.rcParams.update({'font.size': 8})
    ax.bar_label(rect, padding=3)
    ax1.bar_label(rect1, padding=3)
    ax2.bar_label(rect2, padding=3)

    fig.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()
    # plt.show()

    fig.savefig(fname = path.replace('csv', 'png', 1))

def listToSeries(to_append, df):
    a_series = pd.Series(to_append, index=df.columns)
    return a_series

def visualize(model, data, ds, train, alg):
    # print('Visualizing:  ')
    dataFeatures = train.to_numpy()
    instanceLabels = []
    for i in range(dataFeatures.shape[0]):
        instanceLabels.append(i)

    scores = model.get_attribute_tracking_scores(np.array(instanceLabels))

    iterations = []
    only_scores = []
    for i in range(len(scores)):
        iterations.append(scores[i][0])
        only_scores.append(scores[i][1])

    sum = []
    for i in range(len(only_scores[0])):
        sum.append(0)
    for i in range(len(only_scores)):
        for j in range(len(only_scores[i])):
            sum[j] += only_scores[i][j]
    # find the average of the sum of scores and determine relatively important features
    average_score_sum = 0
    for i in range(len(sum)):
        average_score_sum += sum[i]
    average_score_sum = average_score_sum / len(sum)

    important_features = []
    unimportant_features = []
    for i in range(len(sum)):
        if (sum[i] >= average_score_sum):
            important_features.append(i)
        else:
            unimportant_features.append(i)
    # finding the average of the relatively important feature and determining the most important features
    average_score_sum = 0
    for i in range(len(sum)):
        if i in important_features:
            average_score_sum += sum[i]
    print(important_features)
    average_score_sum = average_score_sum / len(important_features)
    important_features = []
    for i in range(len(sum)):
        if (sum[i] > average_score_sum):
            important_features.append(i)
        elif i not in unimportant_features:
            unimportant_features.append(i)

    # print(len(iterations))

    for i in range(len(only_scores)):
        temp = []
        for h in range(len(only_scores[i])):
            temp.append(only_scores[i][h])
        for k in range(len(only_scores[i])):
            if k in unimportant_features:
                only_scores[i].remove(temp[k])

    model.export_iteration_tracking_data("iterationData.csv")
    iterationData = pd.read_csv("iterationData.csv")
    # print(iterationData)
    figure_lcs_score, ax = plt.subplots()

    figure_lcs_score.suptitle('LCS Standard Score per Iteration')

    ax.set_xlabel('Iterations')
    ax.set_ylabel("Standard Score")
    ax.plot(iterationData['Iteration'], iterationData['Accuracy (approx)'])
    feature_names = list(ds.columns)
    features = []
    for i in range(len(important_features)):
        features.append(feature_names[important_features[i]])

    # if os.path.isfile('results/resultsOverIterations/instance_scores.png') == False:
    #     plt.savefig('results/resultsOverIterations/lcs_scores_visualization.png')
    # elif os.path.isfile('results/resultsOverIterations/instance_scores.png'):
    #     plt.savefig('results/resultsOverIterations/lcs_scores_visualization1.png')
    if alg == 'lcs':
        plt.savefig('Results/' + str(data).split('/')[1] + '/' + str(data).split('/')[2].replace('.txt', '') + '/Visualizations/lcs_scores_visualization.png')
    else:
        plt.savefig('Results/' + str(data).split('/')[1] + '/' + str(data).split('/')[2].replace('.txt', '') + '/Visualizations/rfilcs_scores_visualization.png')


    figure_lcs_score.clear()

    figure_instance_score, ax1 = plt.subplots()

    ax1.set_xlabel("Instances")
    ax1.set_ylabel("AT Scores")
    ax1.plot(iterations, only_scores)
    figure_instance_score.suptitle(str(data).split('/')[2].replace('.txt', '') + ": Attribute Performance Scores")
    ax1.legend(features)
    # if os.path.isfile('results/resultsOverIterations/instance_scores.png') == False:
    #     plt.savefig('results/resultsOverIterations/instance_scores.png')
    # elif os.path.isfile('results/resultsOverIterations/instance_scores.png'):
    #     plt.savefig('results/resultsOverIterations/instance_scores1.png')
    if alg == 'lcs':
        plt.savefig('results/' + str(data).split('/')[1] +'/'+str(data).split('/')[2].replace('.txt', '') + '/Visualizations/lcs_instance_scores.png')
    else:
        plt.savefig('Results/' + str(data).split('/')[1] + '/' + str(data).split('/')[2].replace('.txt', '') + '/Visualizations/rfilcs_scores_visualization.png')



def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v

def visAcc(path, alg_type, s_path):
    data = pd.read_csv(path, index_col=[0])
    # print(data.head(10))
    if (alg_type == 'lcs'):
        len = int(data.shape[0] / 2)
        # df = data.iloc[:len, :]
        df = copy.deepcopy(data)
        # name = df.iloc[0, 0]
        for i in range(6):
            toDrop = ['DF_Name', 'CompTime', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 'ROC_AUC']
            d = toDrop[i + 2]
            toDrop.remove(d)
            df1 = df.drop(columns=toDrop)
            ls = np.arange(len).tolist()
            ls1 = np.arange(len).tolist()
            for k in range(0, len):
                val = df1.iloc[k, 1]
                ls[k] =  val
                ls1[k] = df1.iloc[k, 0]

            fig, ax = plt.subplots(figsize=(10, 8))

            plt.plot(ls1, ls)
            # ax = sb.lineplot(x=ls1, y=ls, palette='Blues', )
            # ax = sb.heatmap(a, annot=True, fmt=".2f", cmap='Blues', cbar_kws={"shrink": .8}, xticklabels = [100, 1000, 5000, 10000], yticklabels = [1, 5, 10], ax = ax)

            ax.set_ylabel(d)
            ax.set_xlabel('nIterations')
            # print(data.iloc[0,0])
            print(data.head(10))
            fig.suptitle(data.iloc[0, 0] + '   LCS   ' + str(d))
            plt.savefig('Results/' + str(path).split('/')[1] + '/' + s_path + '/Visualizations/lcs_linePlt' + str(i) + '.png')

            # df = data.iloc[len:, :]
            # # name = df.iloc[0, 0]
            # for i in range(6):
            #     toDrop = ['DF_Name', 'CompTime', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 'ROC_AUC']
            #     d = toDrop[i + 2]
            #     toDrop.remove(d)
            #     df1 = df.drop(columns=toDrop)
            #     ls = np.arange(len).tolist()
            #     ls1 = np.arange(len).tolist()
            #     for k in range(0, len):
            #         val = df1.iloc[k, 1]
            #         ls[k] = val
            #         ls1[k] = df1.iloc[k, 0]
            #
            #     fig, ax = plt.subplots(figsize=(10, 8))
            #
            #     plt.plot(ls1, ls)
            #     # ax = sb.lineplot(x=ls1, y=ls, palette='Blues', )
            #     # ax = sb.heatmap(a, annot=True, fmt=".2f", cmap='Blues', cbar_kws={"shrink": .8}, xticklabels = [100, 1000, 5000, 10000], yticklabels = [1, 5, 10], ax = ax)
            #
            #     ax.set_ylabel(d)
            #     ax.set_xlabel('nIterations')
            #     fig.suptitle(data.iloc[len, 0] + '   LCS   ' + str(d))
            #     # plt.savefig('results/resultsOverIterations/lcs_linePlt' + str(i + 6) + '.png')
            #     plt.savefig('Results/' + s_path + '/Visualizations/lcs_linePlt' + str(i + 6) + '.png')

    if (alg_type == 'rfLCS'):
        len = int(data.shape[0] / 2)
        # df = data.loc[:len, : ]
        df = copy.deepcopy(data)
        for i in range(6):
            toDrop = ['DF_Name', 'CompTime', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 'ROC_AUC']
            d = toDrop[i + 2]
            toDrop.remove(d)
            df1 = df.drop(columns=toDrop)
            a = np.empty(shape=(3, 4))
            count = 0

            for l in range(0,4):
                for k in range (0,4):
                    a[l][k] = df1.iloc[count, 2]
                    count += 1

            fig, ax = plt.subplots(figsize=(10, 8))

            # ax = sb.heatmap(a, annot=True, fmt=".2f", cmap='Blues', cbar_kws={"shrink": .8}, xticklabels = [100, 1000, 5000, 10000], yticklabels = [1, 5, 10], ax = ax)
            ax.set_ylabel('nTrees')
            ax.set_xlabel('nIterations')
            fig.suptitle(data.iloc[0,0] + '   RFLCS   ' + str(d))
            # plt.savefig('results/resultsOverIterations/rfi_lcs_heatmap_iter' + str(i) + '.png')
            plt.savefig('Results/' + str(path).split('/')[1] + '/' + s_path + '/Visualizations/rfi_lcs_heatmap_iter' + str(i) + '.png')

        # df.head()
        # df = data.loc[len:, : ]
        # for i in range(6):
        #     toDrop = ['DF_Name', 'CompTime', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 'ROC_AUC']
        #     d = toDrop[i + 2]
        #     toDrop.remove(d)
        #     df1 = df.drop(columns=toDrop)
        #     a = np.empty(shape=(3, 4))
        #     count = 0
        #
        #     for l in range(0,3):
        #         for k in range (0,4):
        #             a[l][k] = df1.iloc[count, 2]
        #             count += 1
        #
        #     fig, ax = plt.subplots(figsize=(10, 8))
        #
        #     ax = sb.heatmap(a, annot=True, fmt=".2f", cmap='Blues', cbar_kws={"shrink": .8}, xticklabels = [100, 1000, 5000, 10000], yticklabels = [1, 5, 10], ax = ax)
        #     ax.set_ylabel('nTrees')
        #     ax.set_xlabel('nIterations')
        #     fig.suptitle(data.iloc[len,0] + '   RFLCS   ' + str(d))
        #     # plt.savefig('results/resultsOverIterations/rfi_lcs_heatmap_iter' + str(i+6) + '.png')
        #     # print('results/' + str(path).split('/')[2].replace('.txt', '') + '/Visualizations/rfi_lcs_heatmap_iter' + str(i+6) + '.png')
        #     plt.savefig('Results/' + s_path + '/Visualizations/rfi_lcs_heatmap_iter' + str(i+6) + '.png')



