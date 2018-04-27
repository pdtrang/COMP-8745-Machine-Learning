import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import numpy as np
import time

def generateRandomData(numTrainingInstances,numFeatures):
    data = np.random.rand(numTrainingInstances, numFeatures)
    labels = np.random.randint(0,2,numTrainingInstances)
    return (data,labels)


def fiveFoldCrossValidation(model,data,labels):
    kf = KFold(n_splits=5)
    clf = model()
    for train_indices, test_indices in kf.split(data):
        clf.fit(data[train_indices], labels[train_indices])
        predicted = clf.predict(data[test_indices])


def varyNumFeatures():
    nbTimes = []
    knnTimes = []
    logRegTimes = []
    for num in range(5,35,5):
        dataAndLabels = generateRandomData(1000,num)
        data = dataAndLabels[0]
        labels = dataAndLabels[1]
        #
        start = time.time()
        fiveFoldCrossValidation(GaussianNB,data,labels)
        end = time.time()
        timeNB = end - start
        nbTimes.append(timeNB)
        #
        start = time.time()
        fiveFoldCrossValidation(LogisticRegression, data, labels)
        end = time.time()
        timeKNN = end - start
        knnTimes.append(timeKNN)
        #
        start = time.time()
        fiveFoldCrossValidation(KNeighborsClassifier, data, labels)
        end = time.time()
        timeLogReg = end - start
        logRegTimes.append(timeLogReg)
    numFeatures = range(5,35,5)
    line1, = plt.plot(numFeatures,nbTimes)
    line2, = plt.plot(numFeatures,knnTimes)
    line3, = plt.plot(numFeatures,logRegTimes)
    plt.legend(handles=[line1,line2,line3],labels=["Naive Bayes","K-NN","Logistic Regression"])
    plt.ylabel("Time in seconds")
    plt.xlabel("Num features")
    plt.show()

varyNumFeatures()

def varyNumTrainingInstances():
    nbTimes = []
    knnTimes = []
    logRegTimes = []
    numTrainingInstances = [100, 500, 1000, 10000, 25000]
    for num in numTrainingInstances:
        dataAndLabels = generateRandomData(num,10)
        data = dataAndLabels[0]
        labels = dataAndLabels[1]
        #
        start = time.time()
        fiveFoldCrossValidation(GaussianNB,data,labels)
        end = time.time()
        timeNB = end - start
        nbTimes.append(timeNB)
        #
        start = time.time()
        fiveFoldCrossValidation(LogisticRegression, data, labels)
        end = time.time()
        timeKNN = end - start
        knnTimes.append(timeKNN)
        #
        start = time.time()
        fiveFoldCrossValidation(KNeighborsClassifier, data, labels)
        end = time.time()
        timeLogReg = end - start
        logRegTimes.append(timeLogReg)
    line1, = plt.plot(numTrainingInstances,nbTimes)
    line2, = plt.plot(numTrainingInstances,knnTimes)
    line3, = plt.plot(numTrainingInstances,logRegTimes)
    plt.legend(handles=[line1,line2,line3],labels=["Naive Bayes","K-NN","Logistic Regression"])
    plt.ylabel("Time in seconds")
    plt.xlabel("Num training instances")
    plt.show()

varyNumTrainingInstances()


