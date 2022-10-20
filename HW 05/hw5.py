import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # sys.argv[1] #this contains the first argument as string
    with open(sys.argv[1], encoding="UTF8") as csvfile:
        toy = csv.DictReader(csvfile)

        years = []  # x values
        days = []  # y values
        for data in toy:
            years.append(data['year'])
            days.append(int(data['days']))

    years = np.array(years)
    days = np.array(days)

    plt.plot(years, days)
    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')

    # x-axis lable (Year) and y-axis lable (Number of frozen days)
    plt.savefig("plot.jpg")

    # Q3a
    print('Q3a:')
    X = np.ones(shape=(len(years), 2), dtype=np.int64)
    for i in range(len(years)):
        X[i, 1] = years[i]

    print(X)

    # Q3b
    print('Q3b:')
    Y = []
    for i in days:
        Y.append(i)
    Y = np.array(Y)
    print(Y)

    # Q3c
    # Z = X^T * X
    print('Q3c:')
    Z = np.dot(np.transpose(X), X)
    print(Z)

    # Q3d
    # I = inverse of Z = (X^T * X)^-1
    print('Q3d:')
    I = np.linalg.inv(Z)
    print(I)

    # Q3e
    # PI = I * X^T = (X^T * X)^-1 * X^T
    print('Q3e:')
    PI = np.dot(I, np.transpose(X))
    print(PI)

    # Q3f
    # hat_beta = PI*Y = ((X^T)*X)^(-1)*(X^T)*Y
    print('Q3f:')
    hat_beta = np.dot(PI, Y)
    print(hat_beta)

    # Q4
    y_test = hat_beta[0] + hat_beta[1] * 2021
    print('Q4: ' + str(y_test))

    # Q5
    if hat_beta[1] > 0:
        symbol = '>'
    elif hat_beta[1] < 0:
        symbol = '<'
    else:
        symbol = '='
    print('Q5a: ' + symbol)
    print('Q5b: The sign >, <, or = indicates wheter the Mendota ice days have been increased, decreased, or the same as the previous year. For example, If the sign is <, then the ice days on Mendota decreased.')

    # Q6
    x_asterisk = -hat_beta[0] / hat_beta[1]
    print('Q6a: ' + str(x_asterisk))
    print('Q6b: x_asterisk means the days after the data that we have. 0 = βˆ0 + βˆ1x^∗ means the intersection between y (snowy days) = 0 and x axis (year). The result showed 2455.58, and it means when we are in 2456 year, there will be no snowy day')
