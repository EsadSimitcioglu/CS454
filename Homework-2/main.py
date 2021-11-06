import csv
import math
import matplotlib.pyplot as plt

class FlowerPair:
    def __init__(self,value,guessName,realName):
        self.value = value
        self.guessName = guessName
        self.realName = realName

    def __repr__(self):
        return repr((self.value, self.name))

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


class1 = Flower("Iris-setosa")
class2 = Flower("Iris-versicolor")
class3 = Flower("Iris-virginica")


def nearestMeanFinder():
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

    with open("testing.csv") as csv_file:
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

def kNearestNeighbor():
    matrixforK1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    matrixforK3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    matrixforK5 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    matrixCount = matrixforK1, matrixforK3, matrixforK5
    resultValues = []
    resultNames = []
    row = -1

    with open("testing.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        isHeader = True
        for testRow in csv_reader:
            if isHeader:
                isHeader = False
                continue
            row+=1
            resultValues.append([])
            resultNames.append([])
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


        k = 1
        row = 0
        selectedValue = 0



        for matrix in matrixCount:
            if k == 1:
                guessName = resultValues[row][0].guessName
                selectedValue = resultValues[row][0].value
                realName = resultValues[row][0].realName
            elif k == 3:
                nameCount = 1
                selectedName = resultValues[row][0].name
                for i in range(0,k):
                    if(resultValues[row][i].name == selectedName):
                        nameCount += 1



            elif k == 5:
                print(5)


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





#nearestMeanFinder()
kNearestNeighbor()

