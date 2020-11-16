from matminer.datasets import load_dataset
# df = load_dataset("matbench_expt_is_metal") # 4k
df = load_dataset("matbench_mp_is_metal") # 100k
print(df)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=18012019)
print(kf)
