import matplotlib.pyplot as plt
import numpy as np

ele_cnt_pair_lst = [('O', 52025), ('Li', 17724), ('P', 12583), ('Mn', 10231), ('Fe', 9123), ('S', 8614), ('F', 8580), ('Co', 8002), ('H', 7585), ('Cu', 7281), ('Si', 7201), ('Mg', 6996), ('V', 6697), ('Na', 6544), ('Ni', 6351), ('K', 5527), ('Al', 5523), ('Ba', 5351), ('C', 5159), ('B', 5083), ('Ca', 5070), ('Se', 4942), ('Ti', 4918), ('N', 4882), ('Cl', 4525), ('Sr', 4447), ('Cr', 4444), ('Zn', 4322), ('Ge', 4249), ('Sn', 4117), ('La', 3987), ('Sb', 3804), ('Ga', 3788), ('Te', 3668), ('Mo', 3586), ('Rb', 3342), ('Y', 3294), ('Bi', 3287), ('Cs', 3242), ('In', 3208), ('As', 3024), ('Nb', 3008), ('Ag', 2938), ('W', 2876), ('Ce', 2513), ('Cd', 2512), ('Nd', 2470), ('Pd', 2386), ('I', 2369), ('Zr', 2325), ('Pb', 2257), ('Sm', 2216), ('Br', 2212), ('Pr', 2209), ('Tl', 2188), ('Rh', 1995), ('Er', 1929), ('Ta', 1927), ('Au', 1891), ('U', 1890), ('Dy', 1886), ('Ho', 1872), ('Ru', 1835), ('Pt', 1817), ('Tb', 1741), ('Sc', 1710), ('Yb', 1625), ('Hg', 1605), ('Ir', 1556), ('Eu', 1541), ('Tm', 1503), ('Lu', 1446), ('Gd', 1426), ('Hf', 1354), ('Re', 938), ('Th', 874), ('Be', 790), ('Os', 778), ('Pm', 459), ('Tc', 373), ('Ac', 256), ('Np', 251), ('Pu', 248), ('Pa', 219)]

# Fixing random state for reproducibility
np.random.seed(19680801)

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

plt.rcdefaults()
fig, ax = plt.subplots()
ax1 = ax.twinx()

# Example data
eles = []
eles_0 = []
eles_1 = []
cnts = []

for pr in ele_cnt_pair_lst:
    eles.append(pr[0])
    cnts.append(pr[1])

for i in range(len(eles)):
    if i%2 == 0:
        eles_0.append(eles[i])
        eles_1.append(' ')
    else:
        eles_1.append(eles[i])
        eles_0.append(' ')

y_pos = np.arange(len(eles))

ax.barh(y_pos, cnts, align='center')

ax.set_yticks(y_pos)
ax1.set_yticks(y_pos)
ax.set_yticklabels(eles_0)
ax1.set_yticklabels(eles_1)
ax.invert_yaxis()  # labels read top-to-bottom
ax1.invert_yaxis()  # labels read top-to-bottom

ax1.set_ylim(ax.get_ylim())

ax.set_xlabel('Count', font_axis)
# ax.set_title('How fast do you want to go today?')

plt.subplots_adjust(bottom=0.08, right=0.95, left=0.06, top=0.99)
plt.show()


