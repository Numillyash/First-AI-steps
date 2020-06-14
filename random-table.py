from Vectors import *
import random
from random import choice
import csv

myFile = open('tit/answer.csv', 'w', newline='')
with myFile:
    writer = csv.writer(myFile, delimiter=',')
    writer.writerow(['PassengerId', 'Survived'])
    for k, v in get_users_task().items():
        writer.writerow([k, choice([0, 1])])
