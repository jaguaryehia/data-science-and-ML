import pandas as pd
import nltk
import random
import sys
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

names = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
         'physician-fee-freeze'
    , 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
         'mx-missile', 'immigration'
    , 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports'
    , 'export-administration-act-south-africa']

dataset = pd.read_csv('house-votes-84.data', names=names)
# print(dataset['Class Name'].head())

# replacement of missing values with major value
for i in names:
    col = dataset[i]
    most_freq = max(nltk.FreqDist(col))
    # print(most_freq,i)
    # print(col[2],i)
    for j in range(0, len(col)):
        if (col[j] == '?'):
            col[j] = most_freq
    dataset[i] = col

# print(dataset.head())

feature_matrix = ['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
                  'physician-fee-freeze'
    , 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                  'mx-missile', 'immigration'
    , 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports'
    , 'export-administration-act-south-africa']
# print(dataset[feature_matrix])

# map dataset to 0 and 1
for i in names:
    col = dataset[i]
    for j in range(0, len(col)):
        if (col[j] == 'n'):
            col[j] = 0
        elif (col[j] == 'y'):
            col[j] = 1
    dataset[i] = col

# print(dataset)

# Split dataset into training set and test set
X = dataset[feature_matrix]
y = dataset['Class Name']

test_size = [0.2, 0.3, 0.4, 0.5]

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=8 ) 

      #BUILD THE MODEL
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train,y_train)


y_predicted=tree_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_predicted))
treeObj = tree_model.tree_
print (treeObj.node_count)
'''
each_size_accuracy = []
each_size_sTree = []
each_seed = []
all_seeds = []
random_state = [[8, 14, 25, 28, 31], [14, 17, 20, 25, 32], [3, 17, 20, 26, 29], [3, 9, 14, 17, 18]]
original_stdout = sys.stdout
resultFile = open('results.txt', 'w')
print("results of the expriment..")
sys.stdout = resultFile
for i in range(0, 4):
    seeds_accuracy = []
    seeds_sizes = []
    seeds = []
    for j in range(0, 5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size[i],
                                                            random_state=random_state[i][j])

        # BUILD THE MODEL
        tree_model = DecisionTreeClassifier()
        tree_model.random_state = random_state[i][j]
        tree_model.fit(X_train, y_train)

        y_predicted = tree_model.predict(X_test)
        # Model Accuracy, how often is the classifier correct?
        print("size: ", test_size[i], " seed: ", random_state[i][j])
        print("Accuracy:", metrics.accuracy_score(y_test, y_predicted))
        treeObj = tree_model.tree_
        print("size of tree:", treeObj.node_count)
        # each_seed.append([metrics.accuracy_score(y_test, y_predicted),treeObj.node_count,test_size[i],random_state[j]])
        seeds_accuracy.append(metrics.accuracy_score(y_test, y_predicted))
        seeds_sizes.append(treeObj.node_count)
        seeds.append(random_state[i][j])
    print("min acuuraccy:", min(seeds_accuracy))
    print("max accuraccy :", max(seeds_accuracy))
    print("average accuraccy:", sum(seeds_accuracy) / len(seeds_accuracy))
    print("min tree size:", min(seeds_sizes))
    print("max tree size:", max(seeds_sizes))
    print("average tree size:", sum(seeds_sizes) / len(seeds_sizes))
    each_size_accuracy.append([min(seeds_accuracy), max(seeds_accuracy)])
    each_size_sTree.append([min(seeds_sizes), max(seeds_sizes)])
    all_seeds.append(seeds)
# print(each_size_accuracy)
# print(each_size_sTree)

resultFile.close()
sys.stdout = original_stdout
print("done")
