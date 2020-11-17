import numpy as np
np.random.seed(18012019)
from sklearn.metrics import roc_auc_score
from matminer.datasets import load_dataset
import pymatgen.io.ase as pymatgen_io_ase
from sklearn.model_selection import KFold, StratifiedKFold
from itertools import combinations
import time

############# (down)load the data ###############
# df = load_dataset("matbench_expt_is_metal") # 4k, ROC-AUC 0.92, input is composition, not structure...
df = load_dataset("matbench_mp_is_metal") # 100k, 7.2% mem on inode01, ROC-AUC 0.977
# print(df.values.shape)
print(df.head(5))

############# test ROC_AUC value ###########
is_metal = df.values[:,1:2]
is_metal = is_metal.flatten()
rand_array = np.random.rand(is_metal.shape[0])
ret1 = roc_auc_score(is_metal.astype(int), rand_array.astype(float))
ret2 = roc_auc_score(is_metal.astype(int), is_metal.astype(float))
print(ret1, ret2)

############## FUNCTIONS ##################
def cvt_pymatgen2ase(strus):
    ret = []
    for i in range(strus.shape[0]):
        atoms = pymatgen_io_ase.AseAtomsAdaptor.get_atoms(strus[i][0])
        ret.append([atoms])
    return np.array(ret)

def check_element_count(X):
    d = {}
    d_uniq = {}
    for i in range(X.shape[0]):
        d_tmp = {}
        ats = X[i][0]
        for at in ats:
            if at.symbol in d:
                d[at.symbol] += 1
            else:
                d[at.symbol] = 1
            if at.symbol in d_tmp:
                d_tmp[at.symbol] += 1
            else:
                d_tmp[at.symbol] = 1
        for (k, v) in d_tmp.items():
            if k in d_uniq:
                d_uniq[k] += 1
            else:
                d_uniq[k] = 1
    
    print(len(d), d)
    print(len(d_uniq), sorted(d_uniq.items(), key=lambda d: d[1], reverse=True))

def check_clst_count(X, ele_lst, min_cnt, max_cnt, max_length):
    # find and sort all keys
    print(X.shape)
    total_clst_dict = {}
    # for i in range(X.shape[0]):
    for i in range(10):
        if i%10 == 0:
            print(time.asctime( time.localtime(time.time()) ), i)
        ele_idx_lst = []
        for j in range(len(X[i][0])):
            if X[i][0][j].symbol in ele_lst:
                ele_idx_lst.append(j)
        for at_cnts in range(min_cnt, max_cnt):
            for idx_tup in list(combinations(ele_idx_lst, at_cnts)):
                clst_pair_lst = [] # value of a cluster
                for idx_pair in list(combinations(list(idx_tup), 2)):
                    d = X[i][0].get_distance(idx_pair[0], idx_pair[1], mic=True)
                    if d > max_length:
                        break
                    else:
                        clst_pair_lst.append((''.join(
                            sorted([X[i][0][idx_pair[0]].symbol, X[i][0][idx_pair[1]].symbol])
                            ), d))
                else:
                    clst_hash = ''.join(sorted(X[i][0][k].symbol for k in list(idx_tup))) # key of a cluster
                    if clst_hash in total_clst_dict:
                        total_clst_dict[clst_hash].append(clst_pair_lst)
                    else:
                        total_clst_dict[clst_hash] = [clst_pair_lst]
    for (k, v) in sorted(total_clst_dict.items(), key=lambda d: len(d[1]), reverse=True):
        print(k, len(v))
    # all atoms find the value of keys
    pass

def cal_BoCs(X):
    '''
    input is a np.array of ase atoms, output is np.array of BoCs
    '''
    # check_element_count(X)
    ele_candidates = ['O', 'P', 'S', 'F', 'H', 'Si', 'C', 'B', 'N', 'Cl', 'I', 'Br']
    min_cnt = 1
    max_cnt = 3
    max_clst_size = 5
    check_clst_count(X, ele_candidates, min_cnt, max_cnt, max_clst_size)
    # 1. generate whole data
    # 2. clustering
    # 3. calculate BoC of each one
    return X

def extract_features(X):
    X = cvt_pymatgen2ase(X)
    return cal_BoCs(X)

model = None

def train(Xt, Yt):
    '''
    input is BoC (a vector)
    just some pytorch or tf here
    '''
    pass

def predict(Xp):
    '''
    input is list of BoC (a vector), output is list of 0-1 value
    using the trained model to predict
    '''
    rand_array = np.random.rand(Xp.shape[0])
    return rand_array

############ GOOD LUCK!!! #################
if __name__ == '__main__':
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=18012019)
    X = df.values[:,0:1]
    y = df.values[:,1:2].flatten().astype(int)
    # X = extract_features(X)
    ret = []
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        X_train = extract_features(X_train)
        break
        X_test = extract_features(X_test)
        y_train, y_test = y[train_index], y[test_index]
        train(X_train, y_train)
        
        y_pred = predict(X_test)

        ret.append(roc_auc_score(y_test.astype(int), y_pred.astype(float)))
        print('Current results is:', ret)
