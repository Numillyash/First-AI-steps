from Vectors import *
import random
from random import choice
import csv
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
    iteration   - Required  : current iteration (Int)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    length      - Optional  : character length of bar (Int)
    fill        - Optional  : bar fill character (Str)
    printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s ' % (prefix, bar, percent, suffix), end = '')#, end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print("")

def planer(lis: list):
    #a * x + b * y + c * z + d = 0
    max_percent = 0
    a=b=c=d=0
    s1 = time()
    cpun = 0
    for an in np.arange(-2, 2, 0.1):
        for bn in np.arange(-2, 2, 0.1):
            for cn in np.arange(-2, 2, 0.1):
                for dn in np.arange(-2, 2, 0.1):
                    printProgressBar(cpun, 40**4, prefix='Progress:', suffix='Complete', length=100)
                    count = 0
                    for i in lis:
                        if (i[0] == 1 and (an*i[1] + bn*i[2] + cn*i[3] + dn > 0))\
                            or (i[0] == 0 and (an*i[1] + bn*i[2] + cn*i[3] + dn < 0)):
                            count+=1
                    cur_percent = count / len(lis)
                    if (cur_percent > max_percent):
                        max_percent = cur_percent
                        a = an
                        b = bn
                        c = cn
                        d = dn
                    cpun += 1


    printProgressBar(10, 10, prefix='Progress:', suffix='Complete', length=100)
    print(time() - s1)
    print(a,b,c,d, f"Percent = {max_percent}")
    return (a,b,c,d)

def grap(lis: list):

    x = [i[1] for i in lis]
    y = [i[2] for i in lis]
    z = [i[3] for i in lis]
    m = []
    c = []
    for i in lis:
        if i[0] == 0:
            m.append('o')
            c.append('r')
        else:
            m.append('^')
            c.append('b')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c = c[i], marker = m[i])
    xval = np.linspace(2, 11, 100)
    yval = np.linspace(-1, 11, 100)
    x2, y2 = np.meshgrid(xval, yval)


    #-0.1x + 0.1y - 0.2z + 1 = 0
    cort = (-0.1, 0.1, -0.2, 1)#planer(norm_dict.values())
    f = lambda xer, yer: xer * cort[0] + yer * cort[1] + cort[3]
    f2 = lambda xer, yer: xer * 0 + yer * 0
    if(cort[2]!=0):
        z2 = -f(x2, y2)/cort[2]
    else:
        z2 = f2(x2,y2)
    surf = ax.plot_surface(
        # отмечаем аргументы и уравнение поверхности
        x2, y2, z2,
        # шаг прорисовки сетки
        # - чем меньше значение, тем плавнее
        # - будет градиент на поверхности
        rstride=2,
        cstride=2,
        # цветовая схема plasma
        cmap=cm.viridis)
    plt.show()

myFile = open('tit/answer.csv', 'w', newline='')
norm_dict = normalize(get_users_teach())
norm_dict_t = normalize(get_users_task())
#planer(norm_dict.values())
#grap(norm_dict.values())

def checking(d: dict):
    cort = (-0.1, 0.1, -0.2, 1)
    ans_dict = {}
    for k,v in d.items():
        if(cort[0]*v[1]+cort[1]*v[2]+cort[2]*v[3]+cort[3] >= 0):
            ans_dict[k] = [1]
        else:
            ans_dict[k] = [0]
    return ans_dict

with myFile:
    writer = csv.writer(myFile, delimiter=',')
    writer.writerow(['PassengerId', 'Survived'])
    ans = checking(norm_dict_t)
    for k, v in ans.items():
        writer.writerow([k, v[0]])
