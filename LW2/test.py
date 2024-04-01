import random
import math

sizeP = 3
sizeM = 3
sizeQ = 3
nValue = 2
rang = 9
multiplicationTime = 1
additionTime = 1
subtractionTime = 1
comparisonTime = 1
numberOfMultiplications = 0
numberOfAdditions = 0
numberOfSubtractions = 0
numberOfComparisons = 0
consistentTime = 0
parallelTime = 0
averageTime = 0
averageLength = 0
discrepancy = 0

def generate_matrix(first_size, second_size):
    matrix = []
    for i in range(first_size):
        temp_array = []
        for j in range(second_size):
            temp_array.append(round(random.uniform(0, 1), 3))
        matrix.append(temp_array)
    return matrix

def generate_empty_matrix():
    matrix = []
    for i in range(sizeP):
        matrix.append([])
        for j in range(sizeQ):
            matrix[i].append([])
            for k in range(sizeM):
                matrix[i][j].append([])
    return matrix

def reduction(first, second):
    global numberOfSubtractions, numberOfComparisons
    numberOfSubtractions += 1
    numberOfComparisons += 2
    return max(1 - first, second)

def conjunction(first, second):
    global numberOfComparisons
    numberOfComparisons += 2
    return min(first, second)

def single_conjunction(array):
    global numberOfMultiplications
    value = 1
    for num in array:
        value *= num
        numberOfMultiplications += 1
    numberOfMultiplications -= 1
    return value

def single_disjunction(array):
    global numberOfMultiplications, numberOfSubtractions
    value = 1
    for num in array:
        value *= (1 - num)
        numberOfMultiplications += 1
        numberOfSubtractions += 1
    numberOfMultiplications -= 1
    numberOfSubtractions += 1
    return 1 - value

def multiplication(first, second):
    global numberOfAdditions, numberOfSubtractions, numberOfComparisons
    numberOfAdditions += 1
    numberOfSubtractions += 1
    numberOfComparisons += 2
    return max(single_conjunction(first) + single_disjunction(second) - 1, 0)

def calculate():
    global numberOfMultiplications, numberOfAdditions, numberOfSubtractions, numberOfComparisons, parallelTime, consistentTime, averageTime, averageLength, discrepancy
    numberOfMultiplications = 0
    numberOfAdditions = 0
    numberOfSubtractions = 0
    numberOfComparisons = 0
    parallelTime = 0
    consistentTime = 0
    averageTime = 0

    a = generate_matrix(sizeP, sizeM)
    b = generate_matrix(sizeM, sizeQ)
    e = generate_matrix(1, sizeM)
    g = generate_matrix(sizeP, sizeQ)
    d = generate_empty_matrix()
    f = generate_empty_matrix()
    c = generate_empty_matrix()

    for i in range(sizeP):
        for j in range(sizeQ):
            for k in range(sizeM):
                f[i][j][k] = (
                    reduction(a[i][k], b[k][j]) * (2 * e[0][k] - 1) * e[0][k] +
                    reduction(b[k][j], a[i][k]) * (
                        1 + (4 * reduction(a[i][k], b[k][j]) - 2) * e[0][k]
                    ) * (1 - e[0][k])
                )
                numberOfMultiplications += 7
                numberOfAdditions += 2
                numberOfSubtractions += 3

    parallelTime += math.ceil((sizeP * sizeQ * sizeM) / nValue) * (7 * multiplicationTime + 2 * additionTime + 3 * subtractionTime + 3 * (2 * comparisonTime + subtractionTime))
    averageTime += sizeP * sizeQ * sizeM * (7 * multiplicationTime + 2 * additionTime + 3 * subtractionTime + 3 * (2 * comparisonTime + subtractionTime))

    for i in range(sizeP):
        for j in range(sizeQ):
            for k in range(sizeM):
                d[i][j][k] = conjunction(a[i][k], b[k][j])

    parallelTime += math.ceil((sizeP * sizeQ * sizeM) / nValue) * 2 * comparisonTime
    averageTime += sizeP * sizeQ * sizeM * 2 * comparisonTime

    for i in range(sizeP):
        for j in range(sizeQ):
            c[i][j] = (
                single_conjunction(f[i][j]) * (3 * g[i][j] - 2) * g[i][j] + (
                    single_disjunction(d[i][j]) + (4 * (multiplication(f[i][j], d[i][j]) - 3 * single_disjunction(d[i][j])) * g[i][j])
                ) * (1 - g[i][j])
            )

            numberOfMultiplications += 6
            numberOfAdditions += 2
            numberOfSubtractions += 2

    parallelTime += math.ceil((sizeP * sizeQ) / nValue) * (6 * multiplicationTime + 2 * additionTime + 2 * subtractionTime + 3 * ((sizeM - 1) * multiplicationTime + (sizeM + 1) * subtractionTime) + 2 * ((sizeM - 1) * multiplicationTime) + subtractionTime + additionTime + 2 * comparisonTime)
    averageTime += sizeP * sizeQ * (6 * multiplicationTime + 2 * additionTime + 2 * subtractionTime + 3 * ((sizeM - 1) * multiplicationTime + (sizeM + 1) * subtractionTime) + 2 * ((sizeM - 1) * multiplicationTime) + subtractionTime + additionTime + 2 * comparisonTime)

    rang = sizeM * sizeP * sizeQ
    consistentTime = numberOfMultiplications * multiplicationTime + numberOfAdditions * additionTime + numberOfSubtractions * subtractionTime + numberOfComparisons * comparisonTime
    accelerationFactor = consistentTime / parallelTime
    efficiency = consistentTime / (parallelTime * nValue)
    averageLength = math.ceil(averageTime / rang)
    discrepancy = round(parallelTime / averageLength, 3)

calculate()
print("consistentTime:", consistentTime)
print("parallelTime:", parallelTime)
# print("accelerationFactor:", accelerationFactor)
# print("efficiency:", efficiency)
print("averageLength:", averageLength)
print("discrepancy:", discrepancy)