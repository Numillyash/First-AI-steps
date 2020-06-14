import csv
import numpy as np

users_tea = {-1:None}
users_tas = {-1:None}

def get_users_teach():
    users_teach = []
    if(users_tea[list(users_tea.keys())[0]] is None):
        with open('tit/train.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                users_teach.append(row)
        users_teach.pop(0)
        users_tea.pop(list(users_tea.keys())[0])
        for i in users_teach:
            dict_list = [int(i[1]), int(i[2])]
            if(i[4] == "male"):
                dict_list.append(0)
            else:
                dict_list.append(1)
            if(i[5] != ''):
                dict_list.append(int(float(i[5])))
            else:
                dict_list.append(30)
            users_tea[int(i[0])] = dict_list

    return users_tea

def get_users_task():
    users_teach = []
    if(users_tas[list(users_tas.keys())[0]] is None):
        with open('tit/test.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                users_teach.append(row)
        users_teach.pop(0)
        users_tas.pop(list(users_tas.keys())[0])
        for i in users_teach:
            dict_list = [-1, int(i[1])]
            if(i[3] == "male"):
                dict_list.append(0)
            else:
                dict_list.append(1)
            if(i[4] != ''):
                dict_list.append(int(float(i[4])))
            else:
                dict_list.append(30)
            users_tas[int(i[0])] = dict_list

    return users_tas

def normalize(d: dict):
    params_count = len(d[list(d.keys())[0]])
    counts = []
    for i in range(1, params_count):
        min = 1000
        max = -1
        for k, v in d.items():
            if (v[i] > max):
                max = v[i]
            elif (v[i] < min):
                min = v[i]
        counts.append((min, max))
    coeffs = [1]+[10/i[1] for i in counts]
    res = {}
    for k, v in d.items():
        res[k] = [v[i]*coeffs[i] for i in range(len(coeffs))]
    return res

def get_users_teach2():
    users_teach = []
    if(users_tea[list(users_tea.keys())[0]] is None):
        with open('tit/train.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                users_teach.append(row)
        users_teach.pop(0)
        users_tea.pop(list(users_tea.keys())[0])
        for i in users_teach:
            dict_list = [int(i[1]), int(i[2])]
            if(i[4] == "male"):
                dict_list.append(0)
            else:
                dict_list.append(1)
            if(i[5] != ''):
                dict_list.append(int(float(i[5])))
            else:
                dict_list.append(30)
            if (i[6] != ''):
                dict_list.append(int(float(i[6])))
            else:
                dict_list.append(0)
            if (i[9] != ''):
                dict_list.append(float(i[9]))
            else:
                dict_list.append(0)
            if (i[11] != ''):
                if(i[11] == 'S'):
                    dict_list.append(0)
                if (i[11] == 'C'):
                    dict_list.append(1)
                if (i[11] == 'Q'):
                    dict_list.append(2)
            else:
                dict_list.append(0)
            if ("Master" in i[3]):
                dict_list.append(0)
            elif ("Miss" in i[3]):
                dict_list.append(1)
            elif ("Mr" in i[3]):
                dict_list.append(2)
            elif ("Mrs" in i[3]):
                dict_list.append(3)
            else:
                dict_list.append(4)
            users_tea[int(i[0])] = dict_list

    return users_tea

def get_users_task2():
    users_teach = []
    if(users_tas[list(users_tas.keys())[0]] is None):
        with open('tit/test.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                users_teach.append(row)
        users_teach.pop(0)
        users_tas.pop(list(users_tas.keys())[0])
        for i in users_teach:
            dict_list = [-1, int(i[1])]
            if(i[3] == "male"):
                dict_list.append(0)
            else:
                dict_list.append(1)
            if(i[4] != ''):
                dict_list.append(int(float(i[4])))
            else:
                dict_list.append(30)
            if (i[5] != ''):
                dict_list.append(int(float(i[5])))
            else:
                dict_list.append(0)
            if (i[8] != ''):
                dict_list.append(float(i[8]))
            else:
                dict_list.append(0)
            if (i[10] != ''):
                if(i[10] == 'S'):
                    dict_list.append(0)
                if (i[10] == 'C'):
                    dict_list.append(1)
                if (i[10] == 'Q'):
                    dict_list.append(2)
            else:
                dict_list.append(0)

            if ("Master" in i[2]):
                dict_list.append(0)
            elif ("Miss" in i[2]):
                dict_list.append(1)
            elif ("Mr" in i[2]):
                dict_list.append(2)
            elif ("Mrs" in i[2]):
                dict_list.append(3)
            else:
                dict_list.append(4)
            users_tas[int(i[0])] = dict_list

    return users_tas

def get_users_teach3():
    users_teach = []
    if(users_tea[list(users_tea.keys())[0]] is None):
        with open('tit/train.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                users_teach.append(row)
        users_teach.pop(0)
        users_tea.pop(list(users_tea.keys())[0])
        for i in users_teach:
            dict_list = [int(i[1]), int(i[2])]
            if(i[4] == "male"):
                dict_list.append(0)
            else:
                dict_list.append(1)
            users_tea[int(i[0])] = dict_list

    return users_tea

def get_users_task3():
    users_teach = []
    if(users_tas[list(users_tas.keys())[0]] is None):
        with open('tit/test.csv', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                users_teach.append(row)
        users_teach.pop(0)
        users_tas.pop(list(users_tas.keys())[0])
        for i in users_teach:
            dict_list = [-1, int(i[1])]
            if(i[3] == "male"):
                dict_list.append(0)
            else:
                dict_list.append(1)
            users_tas[int(i[0])] = dict_list

    return users_tas


#for k,v in get_users_teach().items():
#    print(k,v)

#for k,v in get_users_task().items():
#    print(k,v)

#normalize(get_users_teach())