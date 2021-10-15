import csv
import math


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
        self.std = math.sqrt(sumOfValues / 3)

    def calculateLikelihhod(self, value):
        powerSide = math.pow(math.e, (-1/2) * math.pow(((value- self.mean))/self.std,2))
        return ((-1)/(self.std * math.sqrt(2*math.pi))) * powerSide


class1 = Class("1")
class2 = Class("2")
class3 = Class("3")

age1 = []
age2 = []
age3 = []


lineCount = 0

with open('training.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    isHeader = True
    for row in csv_reader:
        if isHeader:
            isHeader = False
            continue
        lineCount+=1
        if row[1] == class1.name:
            class1.ages.append(row[0])
        elif row[1] == class2.name:
            class2.ages.append(row[0])
        elif row[1] == class3.name:
            class3.ages.append(row[0])



class1.calculateProps(lineCount)
class2.calculateProps(lineCount)
class3.calculateProps(lineCount)

approx = 26

likelihood1 = class1.calculateLikelihhod(approx)
likelihood2 = class2.calculateLikelihhod(approx)
likelihood3 = class3.calculateLikelihhod(approx)


result1 = (likelihood1 * class1.priors) / ((likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors))
result2 = (likelihood2 * class2.priors) / ((likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors))
result3 = (likelihood3 * class3.priors) / ((likelihood1 * class1.priors) + (likelihood2 * class2.priors) + (likelihood3 * class3.priors))

print(max(result1,result2,result3))




