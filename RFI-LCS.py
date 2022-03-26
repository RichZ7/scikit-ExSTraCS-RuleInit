# Importing necessary packages
import ast
import math

from skExSTraCS import OfflineEnvironment
import h2o
import pandas as pd
from h2o.estimators import H2ORandomForestEstimator
from h2o.tree import H2OTree
from skExSTraCS import ExSTraCS, Classifier
from sklearn import tree
from collections import Counter
import time
import os
import random
h2o.init()


# Defining a class Node
class Node:
    def __init__(self, l_ch, r_ch, f, cond, r_d, l_d, n_id, n_ind, tree):
        self.left_child = l_ch
        self.right_child = r_ch

        self.feature = f
        self.cond = cond
        self.r_path = r_d
        self.l_path = l_d
        self.node_id = n_id
        self.node_index = n_ind

    def toString(self):
        f_str = str(self.feature)
        s = "Left Child (index):   " + str(self.left_child) + "\n" + "Right Child (index):   " + str(
            self.right_child) + "\n" + "Feature:   " + f_str + "\n" + "Right Condition:   " + str(
            self.r_path) + "\n" + "Left Condition:   " + str(self.l_path) + "\n" + "Node Condition:   " + str(
            self.cond) + "\n"
        return s

class Node_c:
    def __init__(self, l_ch, r_ch, f, cond, n_id, n_ind, tree):
        self.left_child = l_ch
        self.right_child = r_ch
        self.feature = f
        self.cond = cond
        self.node_id = n_id
        self.node_index = n_ind

    def toString(self):
        f_str = str(self.feature)
        s = "Left Child (index):   " + str(self.left_child) + "\n" + "Right Child (index):   " + str(
            self.right_child) + "\n" + "Feature:   " + f_str + "\n" +  "\n" + "Node Condition:   " + str(
            self.cond) + "\n"
        return s


# Defining a function to round numbers since Random Forest model makes branches at nodes by determining a cutoff for feature values
# (e.g., if feature 1 > 0.6 then continue to right branch, if feature 1 < 0.6 then continue to left branch
# The rounding function will convert these cutoff values to a discrete 1 or 0 to be used in the rules (since those are the only possible values of the features)
def rounding(number):
    if (number >= 0.5):
        number = 1
    else:
        number = 0
    return number


# Function to check if a given node is a leaf node or not (where leaf node is an 'end' node)
def isLeaf(node):
    if (node.left_child == -1 and node.right_child == -1):
        return True
    else:
        return False


# Recursive function to find paths from the root node to every leaf node
def RootToLeafPaths(node, path, paths, cond, conds, root_pred, tree):
    # base case
    if node == None:
        return
    # include the current node to the path

    path.append(node.feature)

    if (node.cond != None):
        cond.append(node.cond)

    # if a leaf node is found, print the path
    if isLeaf(node):
        paths.append(list(path))
        #         print(paths)
        conds.append(list(cond))
        cond.pop()
        path.pop()
        root_pred.append(node.feature)
        return

    #     recur for the left and right subtree
    if (node.right_child == -1 and node.left_child != -1):
        RootToLeafPaths(tree[node.left_child], path, paths, cond, conds, root_pred, tree)

    elif (node.left_child == -1 and node.right_child != -1):
        RootToLeafPaths(tree[node.right_child], path, paths, cond, conds, root_pred, tree)

    else:
        RootToLeafPaths(tree[node.left_child], path, paths, cond, conds, root_pred, tree)
        RootToLeafPaths(tree[node.right_child], path, paths, cond, conds, root_pred, tree)

    path.pop()

# Recursive function to find paths from the root node to every leaf node
# For Continuous Data, 0 will represent left split and 1, right
def cRootToLeafPaths(node, path, paths, cond, conds, root_pred, tree, condDir, condDirection):
    # base case
    if node == None:
        return
    # include the current node to the path

    path.append(node.feature)

    if (node.cond != None):
        cond.append(node.cond)

    # if a leaf node is found, print the path
    if isLeaf(node):
        paths.append(list(path))
        #         print(paths)
        conds.append(list(cond))
        condDirection.append(list(condDir))
        cond.pop()
        path.pop()
        root_pred.append(node.feature)
        condDir.pop()
        print(len(paths))
        print(len(conds))
        print(len(condDirection))
        return

    #     recur for the left and right subtree
    if (node.right_child == -1 and node.left_child != -1):
        condDir.append(0)
        cRootToLeafPaths(tree[node.left_child], path, paths, cond, conds, root_pred, tree, condDir, condDirection)

    elif (node.left_child == -1 and node.right_child != -1):
        condDir.append(1)
        cRootToLeafPaths(tree[node.right_child], path, paths, cond, conds, root_pred, tree, condDir, condDirection)

    else:
        condDir.append(0)
        cRootToLeafPaths(tree[node.left_child], path, paths, cond, conds, root_pred, tree, condDir, condDirection)
        condDir.append(1)
        cRootToLeafPaths(tree[node.right_child], path, paths, cond, conds, root_pred, tree, condDir, condDirection)

    path.pop()


def removeRoot(paths):
    for i in range(len(paths)):
        paths[i].pop(-1)
    return paths


# put paths and root_pred in lists, write a function for paths with duplicaate attributes
# function should append to paths and root_pred while editing paths with duplicates, takes paths and root_pred as param
# have another variable for paths (as a set)
def duplicateList(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def checkDuplicate(path, cond):
    # return feature and the indice(s) of its duplicate(s)
    # if more than one feature, make it a list, list of lists for duplicate indices
    hasDupe = False
    duplicate_f = []
    duplicate_i = []
    for i in range(len(path)):
        for k in range(len(path)):
            if (i != k):
                if (path[i] == path[k] and path[i] not in duplicate_f):
                    duplicate_f.append(path[k])
    for i in range(len(duplicate_f)):
        locs = duplicateList(path, duplicate_f[i])
        duplicate_i.append(locs)
    #     print(len(duplicate_f))
    #     print(duplicate_f)
    if (len(duplicate_f) > 0):
        hasDupe = True
        return hasDupe, duplicate_f, duplicate_i
    else:
        return hasDupe, None, None


def rmDupe(seq, inds, ind):
    inds1 = list(inds)
    if (ind != 'NA'):
        inds1.pop(ind)
    s = list(seq)
    for i in sorted(inds1, reverse=True):
        del s[i]
    return s


def duplicatesfix(paths, root_pred, conditions):
    paths1 = list(paths)
    #     paths1 = removeRoot(paths)
    preds = list(root_pred)
    conditions1 = list(conditions)
    old = []
    for j in range(len(paths)):
        has_dup, dupes, inds = checkDuplicate(paths[j], conditions[j])
        if (has_dup):
            p = paths[j]
            c = conditions[j]
            pr = root_pred[j]
            old.append(j)
            for a in range(len(dupes)):
                ind = inds[a]
                # loop for each ind of dupe (add new path)
                for l in range(len(ind)):
                    # remove dupes
                    n_p = list(p)
                    n_c = list(c)
                    n_p1 = rmDupe(n_p, ind, l)
                    n_c1 = rmDupe(n_c, ind, l)
                    paths1.append(n_p1)
                    preds.append(pr)
                    conditions1.append(n_c1)
    paths1 = rmDupe(paths1, old, 'NA')
    conditions1 = rmDupe(conditions1, old, 'NA')
    preds = rmDupe(preds, old, 'NA')
    return paths1, preds, conditions1

def cduplicatesfix(paths, root_pred, conditions, condDirections):
    paths1 = list(paths)
    #     paths1 = removeRoot(paths)
    preds = list(root_pred)
    conditions1 = list(conditions)
    condDirections1 = list(condDirections)
    old = []
    for j in range(len(paths)):
        has_dup, dupes, inds = checkDuplicate(paths[j], conditions[j])
        if (has_dup):
            p = paths[j]
            c = conditions[j]
            pr = root_pred[j]
            cC = condDirections[j]
            old.append(j)
            for a in range(len(dupes)):
                ind = inds[a]
                # loop for each ind of dupe (add new path)
                for l in range(len(ind)):
                    # remove dupes
                    n_p = list(p)
                    n_c = list(c)
                    n_cC = list(cC)
                    n_p1 = rmDupe(n_p, ind, l)
                    n_c1 = rmDupe(n_c, ind, l)
                    paths1.append(n_p1)
                    preds.append(pr)
                    conditions1.append(n_c1)
                    condDirections1.append(n_cC)
    paths1 = rmDupe(paths1, old, 'NA')
    conditions1 = rmDupe(conditions1, old, 'NA')
    preds = rmDupe(preds, old, 'NA')
    condDirections1  = rmDupe(condDirections1, old, 'NA')
    return paths1, preds, conditions1, condDirections1


def sort(ok, conds):
    secondOrg = []
    secondOrg_2 = []
    organized_cond = []

    for i in range(len(ok)):
        secondOrg.append(ok[i])

    for i in range(len(ok)):
        ok[i] = sorted(ok[i])

    for i in range(len(secondOrg)):
        temp = []
        for k in range(len(secondOrg[i])):
            temp.append(ok[i].index(secondOrg[i][k]))
        secondOrg_2.append(temp)

    for i in range(len(secondOrg_2)):
        temp = []
        for j in range(len(secondOrg_2[i])):
            temp.append(conds[i][secondOrg_2[i][j]])
        organized_cond.append(temp)

    return ok, organized_cond


def csort(paths, conds, cont_conds):
    secondOrg = []
    secondOrg_2 = []
    organized_cond = []
    org_contCond = []

    for i in range(len(paths)):
        secondOrg.append(paths[i])

    for i in range(len(paths)):
        paths[i] = sorted(paths[i])

    for i in range(len(secondOrg)):
        temp = []
        for k in range(len(secondOrg[i])):
            temp.append(paths[i].index(secondOrg[i][k]))
        secondOrg_2.append(temp)

    for i in range(len(secondOrg_2)):
        temp = []
        for j in range(len(secondOrg_2[i])):
            temp.append(conds[i][secondOrg_2[i][j]])
        organized_cond.append(temp)

    for i in range(len(secondOrg_2)):
        temp = []
        for j in range(len(secondOrg_2[i])):
            temp.append(cont_conds[i][secondOrg_2[i][j]])
        organized_cond.append(temp)

    return paths, organized_cond, cont_conds


# Function to determine the numerosity of rules extracted from RF (numerosity is the number of times each rule occurs)
def getNumerosity(paths, conds, preds):
    paths_2 = list(paths)
    paths_2, conds_2 = sort(paths_2, list(conds))
    numerosities = []
    inds = []
    checked = []
    checked_c = []
    preds1 = []
    for i in range(len(paths)):
        # start at 0 instead of 1 because it will encounter itself
        num = 0
        ind = []
        if (paths[i] not in checked):
            for k in range(len(paths)):
                if (paths_2[i] == paths_2[k] and conds_2[i] == conds_2[k] and preds[i] == preds[k]):
                    num += 1
                    ind.append(k)
            inds.append(ind)
            preds1.append(preds[i])
        if (paths[i] not in checked):
            checked.append(paths_2[i])
            checked_c.append(conds_2[i])
        if (num == 0):
            numerosities.append(1)
        else:
            numerosities.append(num)

    return numerosities, checked, inds, checked_c, preds1

def cgetNumerosity(paths, conds, preds, cont_conds):
    paths_2 = list(paths)
    paths_2, conds_2, cont_conds = csort(paths_2, list(conds), list(cont_conds))
    numerosities = []
    inds = []
    checked = []
    checked_c = []
    preds1 = []
    for i in range(len(paths)):
        # start at 0 instead of 1 because it will encounter itself
        num = 0
        ind = []
        if (paths[i] not in checked):
            for k in range(len(paths)):
                if (paths_2[i] == paths_2[k] and conds_2[i] == conds_2[k] and preds[i] == preds[k]):
                    num += 1
                    ind.append(k)
            inds.append(ind)
            preds1.append(preds[i])
        if (paths[i] not in checked):
            checked.append(paths_2[i])
            checked_c.append(conds_2[i])
        if (num == 0):
            numerosities.append(1)
        else:
            numerosities.append(num)

    return numerosities, checked, inds, checked_c, preds1

def pathList(data, numerosity, n_paths, n_conds, n_preds):
    for i in range(len(n_paths)):
        index = n_paths[i]
        cond = n_conds[i]
        pred = n_preds[i]
        numer = numerosity[i]
        data.append([index, cond, pred, numer])


# Defining a function that will read in the rule_csv and then use it to initialize rules to start LCS with
def RFILCS_Rule_Loading(data_csv, rule_csv, pickle_file, classLabel, number_of_iterations, continuous_attr, cont_conds):
    # Reading in the data and determining number of rules, number of instances in original dataset, and list of attributes in data
    start = time.time()
    data = pd.read_csv(data_csv, delimiter='\t')
    rule_data = pd.read_csv(rule_csv)
    rule_count = rule_data.shape[0]
    print(rule_count)
    instance_count = data.shape[0]

    attribute_list = list(data)

    rule_accuracy_dict = {}

    # Finding the instances in the correct_set and match_set for each rule
    print('Sorting Rules to Match/Correct set')
    for rule in range(0, rule_count):
        print('Rule :' +str(rule))
        match_set = []
        correct_set = []
        attribute_index_string = rule_data.iloc[rule]['Attribute Index']
        attribute_index_list = ast.literal_eval(attribute_index_string)
        # print('AT Index List:')
        # print(attribute_index_list)
        condition_string = rule_data.iloc[rule]['Condition']
        condition_list = ast.literal_eval(condition_string)
        # print('Cond List')
        # print(condition_list)
        for instance in range(0, instance_count):
            match = True

            if continuous_attr:
                # state = data.loc[instance,:][:-1]
                # cont_conds = getCondlist(state=state, cond_list=condition_list)
                # for i in range(len(cont_conds)):
                #     specifiedIndex = attribute_index_list[i]
                #     instanceValue = state[specifiedIndex]
                #     if instanceValue == None:
                #         match = False
                #     elif cont_conds[i][0] < instanceValue < cont_conds[i][1]:
                #         pass
                #     else:
                #         match = False
                for i in range(0, len(attribute_index_list)):
                    if cont_conds[rule][i] == 0:
                        if data.iloc[instance, attribute_index_list[i]] < condition_list[i]:
                            pass
                        else:
                            match = False
                    else: #if cont_conds val is 1
                        if data.iloc[instance, attribute_index_list[i]] >= condition_list[i]:
                            pass
                        else:
                            match = False
            else:
                for i in range(0, len(attribute_index_list)):

                    if str(data.at[instance, attribute_list[attribute_index_list[i]]]) not in condition_list[i]:
                        match = False

            if match == True:
                match_set.append(instance)

            if match == True and data.iloc[instance]['Class'] == rule_data.iloc[rule]['Class']:
                correct_set.append(instance)

        # Calculating rule accuracy based on correct_set and match_set length
        if len(match_set) > 0:
            rule_accuracy_dict[rule] = len(correct_set) / len(match_set)
        elif len(match_set) == 0:
            rule_accuracy_dict[rule] = 0

    newPopSet = []

    # creating a Classifier object (part of ExSTRaCS, each classifier object is a rule in the model) to represent each rule
    # most of the parameters for the Classifier object are place holders, the only important ones needed before pickling the model are attribute index, condition, class, and numerosity
    # each rule will be added to the newPopSet
    for rule in range(0, rule_count):
        dummymodel = ExSTraCS()
        newClassifier = Classifier(dummymodel)
        attribute_index_string = rule_data.iloc[rule]['Attribute Index']
        attribute_index_list = ast.literal_eval(attribute_index_string)
        newClassifier.specifiedAttList = attribute_index_list

        # condition_string = rule_data.iloc[rule]['Condition']
        condition_list = ast.literal_eval(condition_string)
        # newClassifier.condition = condition_list
        if continuous_attr:
            threshold_string = rule_data.iloc[rule]['Condition']
            threshold_list = ast.literal_eval(threshold_string)
            condition_list = []
            # print(threshold_list)
            for i in range(0, len(threshold_list)):
                # print(type(rule_data))
                # print(rule_data)

                if cont_conds[rule][i] == 0:
                    min = data[list(data.columns)[attribute_index_list[i]]].min()
                    condition_list.append([min, threshold_list[i]])
                elif cont_conds[rule][i] == 1:
                    max = data[list(data.columns)[attribute_index_list[i]]].max()
                    condition_list.append([threshold_list[i], max])
            # print(condition_string)
            newClassifier.condition = condition_list

        else:
            condition_string = rule_data.iloc[rule]['Condition']
            condition_list = ast.literal_eval(condition_string)
            newClassifier.condition = condition_list

        newClassifier.phenotype = rule_data.iloc[rule]['Class']
        newClassifier.fitness = rule_accuracy_dict[rule]
        newClassifier.accuracy = rule_accuracy_dict[rule]
        newClassifier.numerosity = rule_data.iloc[rule]['Numerosity']

        newClassifier.aveMatchSetSize = 1
        newClassifier.timeStampGA = 0
        newClassifier.initTimeStamp = 0

        newPopSet.append(newClassifier)

    # Determining the dataFeatures (attributes) and dataPhenotypes (class labels) since they're needed for env; env is needed to pickle the model
    dataFeatures = data.drop(classLabel, axis=1).values
    dataPhenotypes = data[classLabel].values
    env = OfflineEnvironment(dataFeatures, dataPhenotypes, dummymodel)

    # Pickling the model into a txt file as specified by the user
    dummymodel = ExSTraCS()
    dummymodel.env = OfflineEnvironment(dataFeatures, dataPhenotypes, dummymodel)
    dummymodel.hasTrained = True
    dummymodel.iterationCount = dummymodel.learning_iterations
    dummymodel.finalMetrics = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, env, newPopSet, newPopSet]
    print('pickling file')
    dummymodel.pickle_model(pickle_file)
    rInputCompTime = time.time() - start
    return rInputCompTime

    # Running LCS (first with just inputted rules, then with inputted rules used to start LCS with a specified number of iterations)
    # model2 = ExSTraCS(learning_iterations = number_of_iterations,nu=10,N=2000,reboot_filename=pickle_file)
    # print("Score with inputted rules")
    # print(model2.score(dataFeatures,dataPhenotypes))
    # model2.fit(dataFeatures,dataPhenotypes)
    # print("Score with LCS after initialization with rules and learning iterations")
    # print(model2.score(dataFeatures,dataPhenotypes))

def RFI_LCS(data_path, rule_csv_path, pickle_file, continuous_attr, classLabel, ntrees_RFparam, max_depth_RFparam,
            min_rows_RFparam, balance_classes_RFparam, seed_RFparam, number_of_iterations):
    # Intaking data
    gametes_df = h2o.import_file(path=data_path)
    if (continuous_attr==False):
        gametes_df = gametes_df.asfactor()
    gametes_df[classLabel] = gametes_df[classLabel].asfactor()
    predictors = predictors = gametes_df.col_names[:-1]
    # Different for each dataset (outcome variable)
    response = classLabel

    # Initializing Random Forest Model
    gametes_drf = H2ORandomForestEstimator(
        ntrees=ntrees_RFparam,
        max_depth=max_depth_RFparam,
        min_rows=min_rows_RFparam,
        balance_classes=balance_classes_RFparam,
        seed=seed_RFparam)
    r = gametes_df.runif()

    # Running Random Forest Model
    train, test = gametes_df.split_frame(ratios=[.8], seed=1234)  # Ratio 1.0 or 0.8

    start = time.time()

    gametes_drf.train(x=predictors,
                      y=response,
                      training_frame=train,
                      validation_frame=test)

    # gametes_drf.download_mojo('drf', get_genmodel_jar=True)

    
    varimp = gametes_drf.varimp(use_pandas=True)
    varimp.to_csv('varimp'+str(ntrees_RFparam)+'.csv')

    # Extracting Rules from RF Model
    all_p = []
    all_c = []
    all_pr = []
    if continuous_attr:
        all_contConds = []

    count = 0

    #5 for experiment og is ntrees_RFparam
    for i in range(ntrees_RFparam):
        tree1 = H2OTree(model=gametes_drf, tree_number=i, plain_language_rules='AUTO')
        t_tree = []
        columns = gametes_df.columns

        count += 1
        print('count:   ' + str(count))

        if continuous_attr:
            for i in range(len(tree1.node_ids)):
                if (tree1.features[i] == None):
                    t_tree.append(
                        Node_c(l_ch=tree1.left_children[i], r_ch=tree1.right_children[i], f=rounding(tree1.predictions[i]),
                            cond=tree1.thresholds[i],
                            n_id=tree1.node_ids[i], n_ind=i, tree=tree1))
                else:
                    t_tree.append(
                        Node_c(l_ch=tree1.left_children[i], r_ch=tree1.right_children[i], f=columns.index(tree1.features[i]),
                            cond=tree1.thresholds[i],
                            n_id=tree1.node_ids[i], n_ind=i, tree=tree1))
        else:
            for i in range(len(tree1.node_ids)):
                if (tree1.features[i] == None):
                    t_tree.append(
                        Node(l_ch=tree1.left_children[i], r_ch=tree1.right_children[i], f=rounding(tree1.predictions[i]),
                            cond=tree1.levels[i], r_d=tree1.right_cat_split[i], l_d=tree1.left_cat_split[i],
                            n_id=tree1.node_ids[i], n_ind=i, tree=tree1))
                else:
                    t_tree.append(
                        Node(l_ch=tree1.left_children[i], r_ch=tree1.right_children[i], f=columns.index(tree1.features[i]),
                            cond=tree1.levels[i], r_d=tree1.right_cat_split[i], l_d=tree1.left_cat_split[i],
                            n_id=tree1.node_ids[i], n_ind=i, tree=tree1))
        # for i in range(5):
        #     print('Node Index:   ' + str(i))
        #     print(t_tree[i].toString())
        p = []
        paths = []
        c = []
        conditions = []
        root_pred = []
        if continuous_attr:
            condDir = []
            cont_conds = []
            rTest = cRootToLeafPaths(t_tree[0], p, paths, c, conditions, root_pred, t_tree, condDir, cont_conds)
        else:
            rTest = RootToLeafPaths(t_tree[0], p, paths, c, conditions, root_pred, t_tree)
        paths = removeRoot(paths)
        if continuous_attr:
            print('Cont Cond Len:   ' + str(len(cont_conds)))
        print('Fixing Duplicates')
        if (continuous_attr):
            n_paths, n_preds, n_conds, n_cConds = cduplicatesfix(paths, root_pred, conditions, cont_conds)
            # n_paths, n_preds, n_conds, n_cConds = cduplicatesfix(n_paths, n_preds, n_conds, n_cConds)
            print(len(n_paths))
            print(len(n_conds))
            print(len(n_cConds))
        else:
            all_contConds = None
            n_paths, n_preds, n_conds = duplicatesfix(paths, root_pred, conditions)
            n_paths, n_preds, n_conds = duplicatesfix(n_paths, n_preds, n_conds)
       
        all_p += n_paths
        all_c += n_conds
        all_pr += n_preds
        if continuous_attr:
            all_contConds += n_cConds
    print('Finding Numerosities')
    numerosities, checked_alr, indsices, checked_conds, new_preds = getNumerosity(all_p, all_c, all_pr)
    
    # Outputting Rule Information to CSV
    print('Outputing Data')
    ruledata = []
    pathList(ruledata, numerosities, checked_alr, checked_conds, new_preds)
    data_frame = pd.DataFrame(ruledata, columns=['Attribute Index', 'Condition', 'Class', 'Numerosity'])
    data_frame.head(10)
    data_frame.to_csv(rule_csv_path, sep=',', index=None)

    # Loading in the rules to LCS and running LCS
    print('Loading Rules to LCS')
    print(all_contConds)
    rCompTime = RFILCS_Rule_Loading(data_path, rule_csv_path, pickle_file, classLabel, number_of_iterations, continuous_attr=continuous_attr, cont_conds = all_contConds)

    # try:
    #     rCompTime = RFILCS_Rule_Loading(data_path, rule_csv_path, pickle_file, classLabel, number_of_iterations)
    # except:
    #     os.remove('ruleTemp.csv')
    #     os.remove('modelTemp.pickle')

    return time.time() - start, h2o.as_list(train, use_pandas=True),h2o.as_list(test, use_pandas=True), rCompTime

