import pandas as pn
from sklearn import preprocessing
import matplotlib as plt
from sklearn import tree
import statistics
import random

house_votes = pn.read_csv(r"E:\Bioninformatics\year4,sem1\Machine Learning and Bioinformatics\Assignments\house-votes-84.data", header=None)
copy = house_votes
outputY = house_votes[0]

features = copy.drop(columns=copy.columns[0])

modeCol = features.mode()
modeColStr = str(modeCol)
modeColSplit = modeColStr.split()
st = int(len(modeColSplit)/2)+1
modeLast = modeColSplit[st:]

# print(modeCol, "\n", modeColStr, "\n", modeColSplit, "\n", modeLast)
i = 0

process = preprocessing.LabelEncoder()
outProcess = preprocessing.LabelEncoder()
for x in features:
    features[x] = features[x].replace({"?": modeLast[i]})
    process.fit(features[x])
    features[x] = process.transform(features[x])
    i = i + 1

outProcess.fit(outputY)
outputY = outProcess.transform(outputY)

dt = tree.DecisionTreeClassifier()

rangeList = [30, 40, 50, 60, 70, 80]
accuracy = []
sizList = []

accuracyX = []
sizListX = []
accuracyt = []
splits = []
# Random 25% splits
for x in range(3):
    counterX = 0
    split = random.randint(1, int(len(features) - (len(features) * 0.25)))
    siz = int(split + len(features) * 0.25)
    # print(split, siz, len(outputY))
    dt = dt.fit(features[split:siz], outputY[split:siz])
    outA = dt.predict(features[siz:])
    outB = dt.predict(features[:split])
    for x in range(len(outputY[siz:])):
        if siz == len(outputY) - 1:
            # print("BROKE")
            break
        elif outA[x] == outputY[siz:][x]:
            counterX += 1
    for x in range(len(outputY[:split])):
        if split == 0:
            # print("0, break")
            break
        if outB[x] == outputY[:split][x]:
            counterX += 1
    accuracyX.append((counterX / (len(outputY[siz:]) + len(outputY[:split])) * 100))
    nodeSize = dt.tree_.node_count
    sizListX.append(nodeSize)

print("Accuracy of random 25% splitting: ", accuracyX)
print("Tree Size of random 25% splitting: ", sizListX)

accStat = []
sizeStat = []
for i in rangeList:
    sizListt = []
    accuracyt = []
    spli = []
    for x in range(5):
        counter = 0
        split1 = random.randint(1, int(len(features) - (len(features) * (i / 100))))
        siz2 = int(split1 + (len(features) * (i / 100)))
        dt = dt.fit(features[:siz2], outputY[:siz2])
        outA = dt.predict(features[siz2:])
        for x in range(len(outputY[siz2:])):
            if outA[x] == outputY[siz2:][x]:
                counter += 1
        accuracyt.append((counter / len(outputY[siz2:])) * 100)
        siz = dt.tree_.node_count
        sizListt.append(siz)
        spli.append(siz2)
    accStat.append(statistics.mean(accuracyt))
    accStat.append(max(accuracyt))
    accStat.append(min(accuracyt))

    sizeStat.append(statistics.mean(sizListt))
    sizeStat.append(max(sizListt))
    sizeStat.append(min(sizListt))

    accuracy.append(max(accuracyt))
    ind = accuracyt.index(max(accuracyt))
    sizList.append(sizListt[ind])
    splits.append(spli[ind])

j = 0
for i in range(0, len(accStat), 3):
    print("Mean of the tree size with different train sizes: ", sizeStat[i])
    print("Min size of the nodes in the tree with different train sizes: ", sizeStat[i+1])
    print("Max size of the nodes in the tree with different train sizes: ", sizeStat[i+2])
    print("Mean of Accuracies with random splits of size ", rangeList[j], ":", accStat[i])
    print("Max of Accuracies with random splits of size ", rangeList[j], ":", accStat[i+1])
    print("Min of Accuracies with random splits of size ", rangeList[j], ":", accStat[i+2], "\n")
    j += 1

index = accuracy.index(max(accuracy))
dt = dt.fit(features[:splits[index]], outputY[:splits[index]])
outA = dt.predict(features[splits[index]:])

plt.pyplot.plot(rangeList, sizList, color='black', linewidth=1, marker='o', markerfacecolor='pink', markersize=8)
plt.pyplot.show()

plt.pyplot.plot(rangeList, accuracy, color='black', linewidth=1, marker='o', markerfacecolor='purple', markersize=8)
plt.pyplot.show()

tree.plot_tree(dt, filled=True)
# fig = plt.pyplot.figure(figsize=(25, 20))
plt.pyplot.show()
