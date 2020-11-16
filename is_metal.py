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
    ############  result 1. total atoms; 2. how many molecule has atom  ######
# 84 {'K': 29710, 'Mn': 47384, 'O': 1137656, 'Cr': 18796, 'Ni': 36289, 'Cs': 14165, 'Rb': 16150, 'As': 18302, 'Si': 55643, 'Sn': 20592, 'Ca': 25046, 'P': 99527, 'Li': 108505, 'Co': 40417, 'Ge': 27322, 'In': 14173, 'Sb': 20168, 'Hf': 6451, 'Mg': 66967, 'Zn': 22981, 'Cd': 12611, 'Cu': 32682, 'Zr': 12038, 'Ga': 21316, 'Y': 13356, 'Ti': 23994, 'La': 19724, 'Al': 42203, 'Mo': 20058, 'Bi': 18867, 'Sr': 22747, 'V': 32658, 'C': 41573, 'Ba': 27804, 'Na': 39560, 'Be': 5162, 'Fe': 54275, 'W': 11501, 'Ag': 11917, 'S': 95551, 'Dy': 7755, 'Ce': 9816, 'Se': 47694, 'Er': 8274, 'B': 49965, 'Te': 23028, 'Sm': 9089, 'Th': 2745, 'Cl': 53073, 'Ta': 9971, 'I': 25243, 'Hg': 8788, 'Sc': 7591, 'Au': 7150, 'F': 119860, 'Rh': 9337, 'Pb': 11415, 'H': 176277, 'Br': 22603, 'Lu': 5593, 'N': 48598, 'Ac': 513, 'Tm': 5861, 'Gd': 5227, 'Eu': 5257, 'Pu': 843, 'Pt': 7767, 'Pd': 10572, 'U': 7145, 'Np': 807, 'Pr': 10281, 'Nb': 15983, 'Os': 2687, 'Ru': 7225, 'Tc': 1118, 'Tb': 7721, 'Yb': 6181, 'Tl': 8661, 'Re': 5019, 'Nd': 11075, 'Ir': 6247, 'Ho': 7922, 'Pm': 840, 'Pa': 326}
# 84 [('O', 52025), ('Li', 17724), ('P', 12583), ('Mn', 10231), ('Fe', 9123), ('S', 8614), ('F', 8580), ('Co', 8002), ('H', 7585), ('Cu', 7281), ('Si', 7201), ('Mg', 6996), ('V', 6697), ('Na', 6544), ('Ni', 6351), ('K', 5527), ('Al', 5523), ('Ba', 5351), ('C', 5159), ('B', 5083), ('Ca', 5070), ('Se', 4942), ('Ti', 4918), ('N', 4882), ('Cl', 4525), ('Sr', 4447), ('Cr', 4444), ('Zn', 4322), ('Ge', 4249), ('Sn', 4117), ('La', 3987), ('Sb', 3804), ('Ga', 3788), ('Te', 3668), ('Mo', 3586), ('Rb', 3342), ('Y', 3294), ('Bi', 3287), ('Cs', 3242), ('In', 3208), ('As', 3024), ('Nb', 3008), ('Ag', 2938), ('W', 2876), ('Ce', 2513), ('Cd', 2512), ('Nd', 2470), ('Pd', 2386), ('I', 2369), ('Zr', 2325), ('Pb', 2257), ('Sm', 2216), ('Br', 2212), ('Pr', 2209), ('Tl', 2188), ('Rh', 1995), ('Er', 1929), ('Ta', 1927), ('Au', 1891), ('U', 1890), ('Dy', 1886), ('Ho', 1872), ('Ru', 1835), ('Pt', 1817), ('Tb', 1741), ('Sc', 1710), ('Yb', 1625), ('Hg', 1605), ('Ir', 1556), ('Eu', 1541), ('Tm', 1503), ('Lu', 1446), ('Gd', 1426), ('Hf', 1354), ('Re', 938), ('Th', 874), ('Be', 790), ('Os', 778), ('Pm', 459), ('Tc', 373), ('Ac', 256), ('Np', 251), ('Pu', 248), ('Pa', 219)]
    pass

def cal_BoCs(X):
    '''
    input is a np.array of ase atoms, output is np.array of BoCs
    '''
    check_element_count(X)
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
