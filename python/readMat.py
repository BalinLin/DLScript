#! /usr/bin/env python3
# coding=utf-8

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter
import scipy.io

if __name__ == "__main__":
    mat = scipy.io.loadmat('metadata.mat')

    for key in mat:
        print(key)
    #     print(len(mat[key]))
    #     print(mat[key])

    # m = 0
    # n = float('inf')
    # for i in mat['recording'][0]:
    #     m = max(m, i)
    #     n = min(n, i)
    #     print(i)
    # print(m, n)

    # print(len(mat['recording'][0]))
    # print(len(mat['frame'][0]))
    # print(len(mat['split'][0]))
    # print(mat['recording'][0])
    # print(mat['recording'][0][0])
    # # print(mat['recordings'])
    # print(mat['recordings'][0][0])

    # i = 0
    # print(len(mat['recordings'][i]))
    # print(mat['recordings'][i])
    # print(len(mat['recording'][i]))
    # print(mat['recording'][i])
    # print(len(mat['person_identity'][i]))
    # print(mat['person_identity'][i])
    # print(len(mat['frame'][i]))
    # print(mat['frame'][i])
    # print(len(mat['gaze_dir']))
    # print(mat['gaze_dir'][i])
    # print(mat['gaze_dir'][1])
    # print(mat['gaze_dir'][2])

    # im = cv2.imread(os.path.join(
    # 'imgs',
    # mat['recordings']['recording'][i],
    # 'head',
    # '%06d' % mat['person_identity'][i],
    # '%06d.jpg' % mat['frame'][i]
    # ))

    for i in range(100, 180):
        if mat['person_identity'][0][i] == 16:
            print(mat['recording'][0][i], mat['person_identity'][0][i], mat['frame'][0][i], mat['gaze_dir'][i])

