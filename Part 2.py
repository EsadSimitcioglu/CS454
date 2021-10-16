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


def confusionMatrice(filename):
    class1 = Class("1")
    class2 = Class("2")
    class3 = Class("3")
    lineCount = 0

    with open("training.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        isHeader = True
        for row in csv_reader:
            if isHeader:
                isHeader = False
                continue
            lineCount += 1
            if row[1] == class1.name:
                class1.ages.append(int(row[0]))
            elif row[1] == class2.name:
                class2.ages.append(int(row[0]))
            elif row[1] == class3.name:
                class3.ages.append(int(row[0]))

    class1.calculateProps(lineCount)
    class2.calculateProps(lineCount)
    class3.calculateProps(lineCount)

    matrix = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        isHeader = True
        for row in csv_reader:
            if isHeader:
                isHeader = False
                continue
            likelihood1 = class1.calculateLikelihhod(int(row[0]))
            likelihood2 = class2.calculateLikelihhod(int(row[0]))
            likelihood3 = class3.calculateLikelihhod(int(row[0]))
            resultOfClass1 = (likelihood1 * class1.priors) / (
                    (likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors))
            resultOfClass2 = ((likelihood2 * class2.priors) / (
                    (likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors)))
            resultOfClass3 = ((likelihood3 * class3.priors) / (
                    (likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors)))

            maximumResult = max(resultOfClass1,resultOfClass2,resultOfClass3)

            if(maximumResult == resultOfClass1):
                if(row[1] == "1"):
                    matrix[0][0] +=1
                elif (row[1] == "2"):
                    matrix[1][0] += 1
                elif (row[1] == "3"):
                    matrix[2][0] += 1
            elif(maximumResult == resultOfClass2):
                if (row[1] == "1"):
                    matrix[0][1] += 1
                elif (row[1] == "2"):
                    matrix[1][1] += 1
                elif (row[1] == "3"):
                    matrix[2][1] += 1
            elif(maximumResult == resultOfClass3):
                if (row[1] == "1"):
                    matrix[0][2] += 1
                elif (row[1] == "2"):
                    matrix[1][2] += 1
                elif (row[1] == "3"):
                    matrix[2][2] += 1

        print(matrix)


confusionMatrice("training.csv")