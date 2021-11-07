import csv
import math
import matplotlib.pyplot as plt

class tempPair:
    def __init__(self,name,value):
        self.value = value
        self.name = name

class FlowerPair:
    def __init__(self,value,guessName,realName):
        self.value = value
        self.guessName = guessName
        self.realName = realName

    def __repr__(self):
        return repr(self.value)

class Flower:
    def __init__(self, name):
        self.name = name
        self.width = []
        self.length = []
        self.mean = [[]]

    def calculateMean(self):
        lengthMean = 0
        widthMean = 0

        for element in self.length:
            lengthMean += element
        lengthMean /= len(self.length)
        self.mean[0].append(lengthMean)

        for element in self.width:
            widthMean += element
        widthMean /= len(self.width)
        self.mean[0].append(widthMean)

    def calculateMeanOclid(self,x,y):
        return math.sqrt(math.pow(self.mean[0][0]-x,2)+math.pow(self.mean[0][1]-y,2))

def calculateMeanOclid(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))

def calculateResultValues(filename):
    row = -1
    resultValues = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        isHeader = True
        for testRow in csv_reader:
            if isHeader:
                isHeader = False
                continue
            row+=1
            resultValues.append([])
            with open("training.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                isHeader = True
                for trainRow in csv_reader:
                    if isHeader:
                        isHeader = False
                        continue
                    tempFlower = FlowerPair(calculateMeanOclid(float(testRow[0]),float(testRow[1]),float(trainRow[0]),float(trainRow[1])),trainRow[2],testRow[2])
                    resultValues[row].append(tempFlower)
                resultValues[row] = (sorted(resultValues[row], key=lambda flowerPair: flowerPair.value))
    return resultValues

class1 = Flower("Iris-setosa")
class2 = Flower("Iris-versicolor")
class3 = Flower("Iris-virginica")

def nearestMeanFinder(filename):
    matrix = [[0, 0, 0], [0, 0, 0],[0, 0, 0]]

    class1 = Flower("Iris-setosa")
    class2 = Flower("Iris-versicolor")
    class3 = Flower("Iris-virginica")

    with open("training.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        isHeader = True
        for row in csv_reader:
            if isHeader:
                isHeader = False
                continue
            if row[2] == class1.name:
                class1.length.append(float(row[0]))
                class1.width.append(float(row[1]))
            elif row[2] == class2.name:
                class2.length.append(float(row[0]))
                class2.width.append(float(row[1]))
            elif row[2] == class3.name:
                class3.length.append(float(row[0]))
                class3.width.append(float(row[1]))

        class1.calculateMean()
        class2.calculateMean()
        class3.calculateMean()

        plt.scatter(class1.length, class1.width, color='r', marker="x")
        plt.scatter(class1.mean[0][0], class1.mean[0][1], color='black', marker="o")
        plt.scatter(class2.length, class2.width, color='g', marker="x")
        plt.scatter(class2.mean[0][0], class2.mean[0][1], color='brown', marker="o")
        plt.scatter(class3.length, class3.width, color='b', marker="x")
        plt.scatter(class3.mean[0][0], class3.mean[0][1], color='y', marker="o")
        plt.legend([class1.name, "Mean " + class1.name, class2.name, "Mean " + class2.name, class3.name, "Mean " + class3.name])
        plt.show()

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        isHeader = True
        for row in csv_reader:
            if isHeader:
                isHeader = False
                continue
            distanceToClass1 = class1.calculateMeanOclid(float(row[0]),float(row[1]))
            distanceToClass2 = class2.calculateMeanOclid(float(row[0]),float(row[1]))
            distanceToClass3 = class3.calculateMeanOclid(float(row[0]),float(row[1]))

            minimumResult = min(distanceToClass1,distanceToClass2,distanceToClass3)

            if (minimumResult == distanceToClass1):
                if (row[2] == class1.name):
                    matrix[0][0] += 1
                elif (row[2] == class2.name):
                    matrix[1][0] += 1
                elif (row[2] == class3.name):
                    matrix[2][0] += 1
            elif (minimumResult == distanceToClass2):
                if (row[2] == class1.name):
                    matrix[0][1] += 1
                elif (row[2] == class2.name):
                    matrix[1][1] += 1
                elif (row[2] == class3.name):
                    matrix[2][1] += 1
            elif (minimumResult == distanceToClass3):
                if (row[2] == class1.name):
                    matrix[0][2] += 1
                elif (row[2] == class2.name):
                    matrix[1][2] += 1
                elif (row[2] == class3.name):
                    matrix[2][2] += 1

        print("The matrix for your data -> ", *matrix, sep="\n")

def kNearestNeighborA(filename):
    guessName = ""
    resultValues = calculateResultValues(filename)

    for i in range(1,6,2):
        matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for row in range(len(resultValues)):
            realName = resultValues[row][0].realName
            if i == 1:
                guessName = resultValues[row][0].guessName
            elif i == 3 or i == 5:
                classCount = tempPair("Iris-setosa",0),tempPair("Iris-versicolor",0),tempPair("Iris-virginica",0)
                for k in range(0,i):
                    for j in classCount:
                        if(j.name == resultValues[row][k].guessName):
                            j.value += 1
                maxTempPair = classCount[0]
                for element in classCount:
                    if(maxTempPair.value < element.value):
                        maxTempPair = element
                guessName = maxTempPair.name

            if (guessName == class1.name):
                if (realName == class1.name):
                    matrix[0][0] += 1
                elif (realName == class2.name):
                    matrix[1][0] += 1
                elif (realName == class3.name):
                    matrix[2][0] += 1
            elif (guessName == class2.name):
                if (realName == class1.name):
                    matrix[0][1] += 1
                elif (realName == class2.name):
                    matrix[1][1] += 1
                elif (realName == class3.name):
                    matrix[2][1] += 1
            elif (guessName == class3.name):
                if (realName == class1.name):
                    matrix[0][2] += 1
                elif (realName == class2.name):
                    matrix[1][2] += 1
                elif (realName == class3.name):
                    matrix[2][2] += 1
        print("The matrix for your " , i , *matrix, sep="\n")

def kNearestNeighborB(filename):
    resultValues = calculateResultValues(filename)
    plotOfX = list()

    for k in range(1,10,2):
        accuracyOfGuess = list()
        for row in range(len(resultValues)):
            if k == 1:
                if resultValues[row][0].guessName == resultValues[row][0].realName:
                    accuracyOfGuess.append(1)
                else:
                    accuracyOfGuess.append(0)
            elif k == 3 or k == 5 or k == 7 or k == 9:
                classCount = tempPair("Iris-setosa", 0), tempPair("Iris-versicolor", 0), tempPair("Iris-virginica", 0)
                for i in range(0, k):
                    for j in classCount:
                        if (j.name == resultValues[row][i].guessName):
                            j.value += 1
                maxTempPair = classCount[0]
                for element in classCount:
                    if (maxTempPair.value < element.value):
                        maxTempPair = element
                if maxTempPair.name == resultValues[row][0].realName:
                    accuracyOfGuess.append(1)
                else:
                    accuracyOfGuess.append(0)
        accuracy = 0.0
        for i in accuracyOfGuess:
            if i == 1:
                accuracy += 1
        accuracy /= len(accuracyOfGuess)
        plotOfX.append(accuracy)

    plt.plot([1,3,5,7,9], plotOfX, color='r', marker="x")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.ylim(0.9,1)
    plt.show()

#nearestMeanFinder("training.csv")
kNearestNeighborA("training.csv")
#kNearestNeighborB("testing.csv")
