import csv
import math
import matplotlib.pyplot as plt


class Class:
    def __init__(self, name):
        self.name = name
        self.ages = []
        self.mean = 0
        self.std = 0
        self.priors = 0

    def calculateProps(self,lengthOfTheData):
        self.calculateMean()
        self.calculateStd()
        self.calculatePrior(lengthOfTheData)

    def calculatePrior(self, lengthOfTheData):
        self.priors = len(self.ages) / lengthOfTheData

    def calculateMean(self):
        sumOfAges = 0
        for element in self.ages:
            sumOfAges += int(element)
        self.mean = sumOfAges / len(self.ages)

    def calculateStd(self):
        sumOfValues = 0
        for element in self.ages:
            sumOfValues += math.pow((int(element) - self.mean), 2)
        self.std = math.sqrt(sumOfValues / len(self.ages))

    def calculateLikelihhod(self, value):
        powerSide = math.exp((-1/2) * math.pow(((value- self.mean))/self.std,2))
        return (1 / (self.std * math.sqrt(2 * math.pi))) * powerSide

def createGraph(filename):
    class1 = Class("1")
    class2 = Class("2")
    class3 = Class("3")
    ages = set()
    lineCount = 0
    class1Likelihood = []
    class2Likelihood = []
    class3Likelihood = []
    class1Posterior = []
    class2Posterior = []
    class3Posterior = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        isHeader = True
        for row in csv_reader:
            if isHeader:
                isHeader = False
                continue
            lineCount += 1
            ages.add((int(row[0])))
            if row[1] == class1.name:
                class1.ages.append(int(row[0]))
            elif row[1] == class2.name:
                class2.ages.append(int(row[0]))
            elif row[1] == class3.name:
                class3.ages.append(int(row[0]))

    class1.calculateProps(lineCount)
    class2.calculateProps(lineCount)
    class3.calculateProps(lineCount)

    for var in ages:
        likelihood1 = class1.calculateLikelihhod(var)
        likelihood2 = class2.calculateLikelihhod(var)
        likelihood3 = class3.calculateLikelihhod(var)
        class1Likelihood.append(likelihood1)
        class2Likelihood.append(likelihood2)
        class3Likelihood.append(likelihood3)
        class1Posterior.append((likelihood1 * class1.priors) / (
                    (likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors)))
        class2Posterior.append((likelihood2 * class2.priors) / (
                    (likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors)))
        class3Posterior.append((likelihood3 * class3.priors) / (
                    (likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors)))

    plt.plot(list(ages), class1Posterior, color='r', linestyle="dashed")
    plt.plot(list(ages), class2Posterior, color='g', linestyle="dashed")
    plt.plot(list(ages), class3Posterior, color='b', linestyle="dashed")

    plt.plot(list(ages), class1Likelihood, color='r')
    plt.plot(list(ages), class2Likelihood, color='g')
    plt.plot(list(ages), class3Likelihood, color='b')

    plt.scatter(class1.ages, [-0.1] * len(class1.ages), color='r', marker="x")
    plt.scatter(class2.ages, [-0.2] * len(class2.ages), color='g', marker="x")
    plt.scatter(class3.ages, [-0.3] * len(class3.ages), color='b', marker="x")

    plt.show()

#createGraph("training.csv")
#createGraph("testing.csv")