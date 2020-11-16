############# (down)load the data ###############
from matminer.datasets import load_dataset
df = load_dataset("matbench_expt_is_metal") # 4k, ROC-AUC 0.92
# df = load_dataset("matbench_mp_is_metal") # 100k, 7.2% mem on inode01, ROC-AUC 0.977
print(df.values.shape)

############# test ROC_AUC value ###########
import numpy as np
np.random.seed(18012019)
from sklearn.metrics import roc_auc_score
is_metal = df.values[:,1:2]
is_metal = is_metal.flatten()
rand_array = np.random.rand(is_metal.shape[0])
ret1 = roc_auc_score(is_metal.astype(int), rand_array.astype(float))
ret2 = roc_auc_score(is_metal.astype(int), is_metal.astype(float))
print(ret1, ret2)

############ k-fold ################# 
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=18012019)
print(kf)
