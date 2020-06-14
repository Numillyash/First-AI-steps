import numpy as np
import warnings
from Vectors import *
warnings.filterwarnings('ignore')
import csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (8,6)
fig = plt.figure()
#%matplotlib inline

import sklearn
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def newline(p1, p2, color=None): # функция отрисовки линии
    #function kredits to: https://fooobar.com/questions/626491/how-to-draw-a-line-with-matplotlib
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], color=color)
    ax.add_line(l)
    return l

def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0],a.shape[1]+1))
    a_extended[:,:-1] = a
    a_extended[:,-1] = int(1)
    return a_extended

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

class CustomSVM(object):

    __class__ = "CustomSVM"
    __doc__ = ''
    def __init__(self, etha=0.01, alpha=0.1, epochs=200):
        self._epochs = epochs
        self._etha = etha
        self._alpha = alpha
        self._w = None
        self.history_w = []
        self.train_errors = None
        self.val_errors = None
        self.train_loss = None
        self.val_loss = None

    def fit(self, X_train, Y_train, X_val, Y_val, verbose=False): #arrays: X; Y =-1,1

        if len(set(Y_train)) != 2 or len(set(Y_val)) != 2:
            raise ValueError("Number of classes in Y is not equal 2!")

        X_train = add_bias_feature(X_train)
        X_val = add_bias_feature(X_val)
        self._w = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])
        self.history_w.append(self._w)
        train_errors = []
        val_errors = []
        train_loss_epoch = []
        val_loss_epoch = []

        for epoch in range(self._epochs):
            printProgressBar(epoch, self._epochs, prefix='Progress:', suffix='Complete', length=100)
            tr_err = 0
            val_err = 0
            tr_loss = 0
            val_loss = 0
            for i,x in enumerate(X_train):
                margin = Y_train[i]*np.dot(self._w,X_train[i])
                if margin >= 1: # классифицируем верно
                    self._w = self._w - self._etha*self._alpha*self._w/self._epochs
                    tr_loss += self.soft_margin_loss(X_train[i],Y_train[i])
                else: # классифицируем неверно или попадаем на полосу разделения при 0<m<1
                    self._w = self._w +\
                    self._etha*(Y_train[i]*X_train[i] - self._alpha*self._w/self._epochs)
                    tr_err += 1
                    tr_loss += self.soft_margin_loss(X_train[i],Y_train[i])
                self.history_w.append(self._w)
            for i,x in enumerate(X_val):
                val_loss += self.soft_margin_loss(X_val[i], Y_val[i])
                val_err += (Y_val[i]*np.dot(self._w,X_val[i])<1).astype(int)
            #if verbose:
            #    print('epoch {}. Errors={}. Mean Hinge_loss={}'\.format(epoch,err,loss))
            train_errors.append(tr_err)
            val_errors.append(val_err)
            train_loss_epoch.append(tr_loss)
            val_loss_epoch.append(val_loss)
        self.history_w = np.array(self.history_w)
        self.train_errors = np.array(train_errors)
        self.val_errors = np.array(val_errors)
        self.train_loss = np.array(train_loss_epoch)
        self.val_loss = np.array(val_loss_epoch)
        printProgressBar(10, 10, prefix='Progress:', suffix='Complete', length=100)

    def predict(self, X:np.array) -> np.array:
        y_pred = []
        X_extended = add_bias_feature(X)
        for i in range(len(X_extended)):
            y_pred.append(np.sign(np.dot(self._w,X_extended[i])))
        return np.array(y_pred)

    def hinge_loss(self, x, y):
        return max(0,1 - y*np.dot(x, self._w))

    def soft_margin_loss(self, x, y):
        return self.hinge_loss(x,y)+self._alpha*np.dot(self._w, self._w)

norm_dict = get_users_teach3()
norm_dict_t = get_users_task3()
data = []
for i in norm_dict.values():
    data.append([i[1:], i[0]])
data2 = []
for i in norm_dict_t.values():
    data2.append([i[1:], i[0]])
iris = load_iris()
X = np.array([i[0] for i in data])
y_s = np.array([i[1] for i in data])
Y = y_s
X2 = np.array([i[0] for i in data2])

pca = PCA(n_components=3)
#X = pca.fit_transform(X)

#X2 = pca.fit_transform(X2)
Y = (Y > 0).astype(int)*2-1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=2020)

svm = CustomSVM(etha=0.01, alpha=0.01, epochs=200)
svm.fit(X_train, Y_train, X, Y)

#print(svm.train_errors) # numbers of error in each epoch
print(svm._w) # w0*x_i[0]+w1*x_i[1]+w2=0


def isPlaneRight(a,b,c,d):
    lis = get_users_teach().values()
    count = 0
    for i in lis:
        if (i[0] == 1 and (a * i[1] + b * i[2] + c * i[3] + d >= 0)) \
                or (i[0] == 0 and (a * i[1] + b * i[2] + c * i[3] + d <= 0)):
            count += 1
    cur_percent = count / len(lis) * 100
    print(f"Percent of good: {cur_percent}%")

w = svm._w

#isPlaneRight(w[0],w[1],w[2],w[3])

y_pred = svm.predict(X)
y_pred[y_pred != Y] = -100 # find and mark classification error
print('Количество ошибок для отложенной выборки: ', (y_pred == -100).astype(int).sum() / len(y_pred) * 100,"%")


d = {-1:'green', 1:'red'}
plt.scatter(X_train[:,0], X_train[:,1], c=[d[y] for y in Y_train])
newline([0,-svm._w[2]/svm._w[1]],[-svm._w[2]/svm._w[0],0], 'blue') # в w0*x_i[0]+w1*x_i[1]+w2*1=0 поочередно
                                                        # подставляем x_i[0]=0, x_i[1]=0
newline([0,1/svm._w[1]-svm._w[2]/svm._w[1]],[1/svm._w[0]-svm._w[2]/svm._w[0],0]) #w0*x_i[0]+w1*x_i[1]+w2*1=1
newline([0,-1/svm._w[1]-svm._w[2]/svm._w[1]],[-1/svm._w[0]-svm._w[2]/svm._w[0],0]) #w0*x_i[0]+w1*x_i[1]+w2*1=-1
plt.show()


plt.plot(svm.train_loss, linewidth=2, label='train_loss')
plt.plot(svm.val_loss, linewidth=2, label='test_loss')
plt.grid()
plt.legend(prop={'size': 15})
plt.show()


def checking():
    ans_dict = {}
    for k,v in zip(norm_dict_t.keys(), y_predict):
        ans_dict[k] = v
    return ans_dict

y_predict = svm.predict(X2)

y_predict = ((y_predict+1)/2).astype(int)


myFile = open('tit/answer2.csv', 'w', newline='')
with myFile:
    writer = csv.writer(myFile, delimiter=',')
    writer.writerow(['PassengerId', 'Survived'])
    ans = checking()
    for k, v in ans.items():
        writer.writerow([k, v])