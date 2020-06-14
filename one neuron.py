from Vectors import *
import random
from random import choice
import csv
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

s=time()

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

myFile = open('tit/answer.csv', 'w', newline='')
norm_dict = get_users_teach()#normalize(get_users_teach())

norm_dict_t = get_users_task()#normalize(get_users_task())


trues = []
for i in norm_dict.values():
    if(i[0] == 0):
        trues.append(-1)
    else:
        trues.append(1)
all_y_truesr = np.array(trues)

dater = []
for i in norm_dict.values():
    dater.append(i[1:]+[1])
datar = np.array(dater)

# Набор точек X:Y
data = []

for i in range(len(dater)):
    data.append([dater[i], all_y_truesr[i]])



a = -0.1
b = 0.1
c = -0.2
d = 1

corta = (a,b,c,d)

def grap(lis: list):

    x = [i[1] for i in lis]
    y = [i[2] for i in lis]
    z = [i[3] for i in lis]
    m = []
    col = []
    for i in lis:
        if i[0] == -1:
            m.append('o')
            col.append('r')
        else:
            m.append('^')
            col.append('b')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c = col[i], marker = m[i])

    xval = np.linspace(2, 11, 100)
    yval = np.linspace(-1, 11, 100)

    x2, y2 = np.meshgrid(xval, yval)

    #-0.5x + 0.5y - 1.0z + 4.5 = 0
    cort = (a,b,c,d)

    f = lambda xer, yer: xer * cort[0] + yer * cort[1] + cort[3]
    f2 = lambda xer, yer: xer * 0 + yer * 0

    if(cort[2]!=0):
        z2 = -f(x2, y2)/cort[2]
    else:
        z2 = f2(x2, y2)
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

def isPlaneRight():
    lis = norm_dict.values()
    count = 0
    for i in lis:
        if (i[0] == 1 and (a * i[1] + b * i[2] + c * i[3] + d > 0)) \
                or (i[0] == 0 and (a * i[1] + b * i[2] + c * i[3] + d < 0)):
            count += 1
    cur_percent = count / len(lis) * 100
    print(f"Percent on good: {cur_percent}%")

# Вывод данных начальной прямой
print(f'Начальная плоскость: {a}*x + {b}*y + {c}*z + {d} = 0')

isPlaneRight()

# класс, который реализует персептрон и его обучение
class TPerceptron:
    def __init__(self, N, cor = None):
        # создать нулевые веса

        self.w = list()
        for i in range(N):
            if(cor == None):
                self.w.append(0)
            else:
                self.w.append(cor[i])
    # метод для вычисления значения персептрона
    def calc(self, x):
        res =  0
        for i in range(len(self.w)):
            res = res + self.w[i] * x[i]
        return res
    # пороговая функция активации персептрона
    def sign(self, x):
        if self.calc(x) >  0:
            return 1
        else:
            return -1
    # обучение на одном примере
    def learn(self, la, x, y):
        # обучаем только, когда результат неверный
        if y * self.calc(x) <= 0:
            for i in range(len(self.w)):
                self.w[i] = self.w[i] + la * y * x[i]
    # обучение по всем данным T - кортеж примеров
    def learning(self, la, T):
        # цикл обучения
        for n in range(10000):
            printProgressBar(n, 10000, prefix='Progress:', suffix='Complete', length=100)
            # обучение по всем набору примеров
            for t in T:
                self.learn(la, t[0], t[1])

        printProgressBar(10, 10, prefix='Progress:', suffix='Complete', length=100)

ler = 0.0001

#perc = TPerceptron(4)

perc = TPerceptron(4)#, corta)

perc.learning(ler, data)

cort2 = tuple(perc.w)

a = cort2[0]
b = cort2[1]
c = cort2[2]
d = cort2[3]

#print(cort2)


# Вывод данных готовой плоскости
print(f'Новая плоскость: {a}*x + {b}*y + {c}*z + {d} = 0')

isPlaneRight()

print(time()-s)

#grap(norm_dict.values())

def checking(di: dict):
    cort = (a,b,c,d)
    ans_dict = {}
    for k,v in di.items():
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