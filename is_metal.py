import numpy as np
np.random.seed(18012019)
from sklearn.metrics import roc_auc_score
from matminer.datasets import load_dataset
import pymatgen.io.ase as pymatgen_io_ase
from sklearn.model_selection import KFold, StratifiedKFold

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

def cal_BoCs(X):
    '''
    input is a np.array of ase atoms, output is np.array of BoCs
    '''
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
    X = extract_features(X)
    ret = []
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        train(X_train, y_train)
        
        y_pred = predict(X_test)

        ret.append(roc_auc_score(y_test.astype(int), y_pred.astype(float)))
        print('Current results is:', ret)
