import clutter as RFI
import testing_job as tj
import h2o
import pandas as pd
h2o.init()
import time
import os
import shutil

#Create Dataframes to store performance results

rf_results = pd.DataFrame(columns=['DF_Name', 'nTrees', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity', 'ROC_AUC'])
lcs_results = pd.DataFrame(columns=['DF_Name', 'nIterations', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity', 'ROC_AUC'])
rf_lcs_noIter_results = pd.DataFrame(columns=['DF_Name', 'nTrees', 'nIterations', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity', 'ROC_AUC'])
rfLCS_results =  pd.DataFrame(columns=['DF_Name', 'nTrees', 'nIterations', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity', 'ROC_AUC'])

#Find names of target Datasets (Should be found in "datasets" subdir)

targets = []
import os
for dir in os.listdir("datasets"):
    if os.path.isdir("datasets/" + dir):
        for file in os.listdir("datasets/"+dir):
            if file.endswith(".txt"):
                targets.append('datasets/' + dir + '/' + file)

print(targets)

# if os.path.isdir('Results'):
#     shutil.rmtree('Results', ignore_errors = True)
os.mkdir('Results')

# For every dataset, run an experiment on the data and record the results
for data_path in targets:
    data = pd.read_csv(data_path, delimiter='\t')
    d = str(data_path).split('/')[2].replace('.txt', '')
    f = str(data_path).split('/')[1]
    print(d)
    #     find if attributes are continous (assumes all are if there is one)
    continous_attr = isinstance(data.iloc[0, 0], float)
    #     get scores for Random Forest and RF_LCS without learning iterations

    os.mkdir('Results/' + f + '/' + d)
    os.mkdir('Results/' + f + '/' + d + '/Visualizations')

    #     tj.runRFLCS(data_path, True, continous_attr, 10, 20000)

    # Run RF and RF_LCS (no lcs iterations)
    for num_trees in [1, 5, 10, 20]:
    # for num_trees in [1, 5]:
        avg_score = [0 for j in range(len(rf_results.columns))]
        for i in range(1):
            curr = tj.runRF(data_path, continous_attr, num_trees)
            avg_score[0] = curr[0]
            for k in range(len(avg_score) - 1):
                avg_score[k + 1] += float(curr[k + 1])
        for x in range(len(avg_score) - 1):
            avg_score[x + 1] = avg_score[i + 1] / 1
        rf_results = rf_results.append(tj.listToSeries(avg_score, rf_results), ignore_index=True)

        avg_score = [0 for j in range(len(rf_lcs_noIter_results.columns))]
        for i in range(1):
            curr = tj.runRFLCS(data_path, False, continous_attr, num_trees, 0)
            avg_score[0] = curr[0]
            for k in range(len(avg_score) - 1):
                avg_score[k + 1] += float(curr[k + 1])
        for x in range(len(avg_score) - 1):
            avg_score[x + 1] = avg_score[x + 1] / 1
        rf_lcs_noIter_results = rf_lcs_noIter_results.append(tj.listToSeries(avg_score, rf_lcs_noIter_results),
                                                             ignore_index=True)

    # Run Lcs
    for nIterations in [100, 1000, 5000, 10000]:
    # for nIterations in [100, 150]:
        avg_score = [0 for j in range(len(lcs_results.columns))]
        for i in range(1):
            curr = tj.runLCS(data_path, nIterations)
            avg_score[0] = curr[0]
            for k in range(len(avg_score) - 1):
                avg_score[k + 1] += float(curr[k + 1])

        for x in range(len(avg_score) - 1):
            avg_score[x + 1] = avg_score[x + 1] / 1
        lcs_results = lcs_results.append(tj.listToSeries(avg_score, lcs_results), ignore_index=True)

    # Run RF_LCS
    # for num_trees in [1, 5]:
    for num_trees in [1, 5, 10, 20]:
        for nIterations in [100, 1000, 5000, 10000]:
        # for nIterations in [100, 150]:
            avg_score = [0 for j in range(len(rfLCS_results.columns))]
            for i in range(1):
                curr = tj.runRFLCS(data_path, True, continous_attr, num_trees, nIterations)
                avg_score[0] = curr[0]
                for k in range(len(avg_score) - 1):
                    avg_score[k + 1] += float(curr[k + 1])

            for x in range(len(avg_score) - 1):
                avg_score[x + 1] = avg_score[x + 1] / 1
            rfLCS_results = rfLCS_results.append(tj.listToSeries(avg_score, rfLCS_results), ignore_index=True)
        tj.plotBar('Results/' + f + '/' + d + '/Visualizations/varimp' + str(num_trees) + '.csv')
    #         tj.plotBar(data_path)


    rf_results.to_csv('Results/' + f + '/' +  d + '/rf_results.csv')
    lcs_results.to_csv('Results/' + f + d + '/lcs_results.csv')
    rf_lcs_noIter_results.to_csv('Results/' + f + '/' + d + '/rf_lcs_noIter_results.csv')
    rfLCS_results.to_csv('Results/' + f + '/' + d + '/rfLCS_results.csv')

    tj.visAcc('Results/' + f + '/' + d + '/lcs_results.csv', 'lcs', d)
    tj.visAcc('Results/' + f + '/' + d + '/rfLCS_results.csv', 'rfLCS', d)

    #Reset Result Dataframes in Memory for Next Run
    rf_results = pd.DataFrame(
        columns=['DF_Name', 'nTrees', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity', 'ROC_AUC'])
    lcs_results = pd.DataFrame(
        columns=['DF_Name', 'nIterations', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity',
                 'ROC_AUC'])
    rf_lcs_noIter_results = pd.DataFrame(
        columns=['DF_Name', 'nTrees', 'nIterations', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity',
                 'ROC_AUC'])
    rfLCS_results = pd.DataFrame(
        columns=['DF_Name', 'nTrees', 'nIterations', 'CompTime', "Accuracy", "F1", "Precision", 'Recall', 'Specificity',
                 'ROC_AUC'])
