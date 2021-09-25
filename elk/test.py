#! /usr/bin/env python3
# coding=utf-8

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
import pandas as pd
from collections import Counter

if __name__ == "__main__":
    df = pd.read_csv('/home/balin/exper/docker-elk/mydata/AnalysisLog_1205_1206_edit.csv', usecols=["UserID", "EventID"])
    f = open('/home/balin/exper/test/test.txt', 'w')
    dfgroup = df.groupby('UserID').groups
    count = 0
    print(len(list(dfgroup)))
    mylist = []
    for i in dfgroup:
        # print(i)
        # print(0 in df.iloc[list(dfgroup[i])]["EventID"].values)
        f.write(str(i))
        s = "\n"
        # for j in list(dfgroup[i]):
        #     s += str(df.iloc[j]["EventID"]) + " "
        
        MyList = list(df.iloc[list(dfgroup[i])]["EventID"].values)
        MyList.sort()
        # print(MyList)

        my_dict = dict(Counter(MyList))

        # print(my_dict)
        for key in my_dict:
            s += str(key) + ":" + str(my_dict[key]) + "    "

        # if 0 not in df.iloc[list(dfgroup[i])]["EventID"].values:
        #     for k in list(dfgroup[i]):
        #         mylist.append(k)

        s += '\n'
        s += "==================\n"
        f.write(s)

        # print("")
        # print("\n================")

    df = df.drop(mylist)
    df.to_csv('/home/balin/exper/test/test.csv', index = 0)