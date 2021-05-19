import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_PATH = os.path.join(ROOT_DIR, 'data/')
out_fname = ['rgbnss_trn', 'rgbnss_tst', 'depth_trn', 'depth_tst']

xtrn_list = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'xtrn_list.txt'))]
xtst_list = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'xtst_list.txt'))]
ytrn_list = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'ytrn_list.txt'))]
ytst_list = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'ytst_list.txt'))]

flist = [xtrn_list, xtst_list, ytrn_list, ytst_list]

data_merged = []
for i in range(len(flist)):
    for f in flist[i]:
        data = np.load(DATA_PATH+f, allow_pickle=True)
        data_merged.append(data)
    data_stack = np.concatenate(data_merged, axis=0)
    np.save(DATA_PATH+'/'+out_fname[i], data_stack, allow_pickle=True)
    data_merged = []
