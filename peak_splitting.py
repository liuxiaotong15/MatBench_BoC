from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 6, 5, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 13, 45, 77, 211, 228, 155, 453, 291, 71, 97, 114, 190, 591, 1148, 1964, 2202, 2645, 2151, 2619, 3788, 4873, 5497, 4942, 3423, 2309, 2653, 3674, 4237, 3878, 3884, 4381, 5070, 5500, 5736, 5633, 5718, 5733, 6001, 5621, 5541, 5621, 5369, 5348, 4740, 5677, 5535, 4875, 4319, 3813, 3527, 3345, 3280, 3486, 3289, 3115, 2978, 2828, 2982, 2589, 2535, 2408, 2450, 2353, 2636, 2245, 2293, 2218, 2175, 2161, 2094, 2051, 2192, 1908, 2151, 2101, 2132, 1845, 1883, 1865, 2015, 2111, 2086, 2182, 2339, 2157, 2323, 2362, 2596, 2756, 2727, 2516, 2849, 2730, 2804, 2621, 2680, 2645, 2862, 2867, 2761, 2536, 2639, 2687, 2688, 2669, 2868, 3151, 3002, 2991, 3081, 2833, 3051, 3038, 2974, 2869, 3130, 2939, 3059, 2890, 2797, 2923, 3209, 3421, 3629, 3538, 3646, 3611, 3648, 3737, 3908, 3958, 4112, 4270, 4263, 4493, 4382, 4349, 4785, 4646, 4428, 4852, 4832, 5277, 5399, 5289, 5234, 5425, 5479, 5122, 5586, 5283, 5430, 5176, 5695, 5802, 5750, 5527, 5632, 4802, 4871, 4286, 4052, 4249, 4275, 3887, 4001, 3920, 4221, 3710, 3583, 3375, 3611, 3534, 3172, 3350, 3483, 3218, 3195, 3117, 2796, 2890, 2935, 2889, 3041, 2921, 2997, 3016, 2959, 2829, 2947, 2827, 2929, 2959, 2907, 3026, 2746, 2934, 2664, 2879, 2687, 2535, 2770, 2671, 2550, 2657, 2523, 2576, 2663, 2833, 2502, 2538, 2457, 2359, 2479, 2539, 2829, 2576, 2404, 2794, 2575, 2599, 2633, 2259, 2288, 2463, 2349, 2382, 2275, 2290, 2134, 2142, 2273, 2322, 2291, 2199, 2219, 2034, 2236, 2219, 2266, 2290, 2198, 2422, 2109, 2222, 2352, 2137, 2188, 2261, 2288, 2288, 2356, 2068, 2147, 2175, 1933, 2108, 1923, 2146, 1866, 2121, 2560, 2256, 2276, 2030, 1881, 2164, 1935, 1848, 1795, 1849, 1880, 1804, 1820, 1835, 1907, 1762, 1766, 1540, 1609, 1681, 1621, 1642, 1561, 1761, 1652, 1731, 1440, 1843, 1691, 1353, 1672, 1594, 1516, 1716, 1591, 1304, 1326, 1336, 1442, 1427, 1585, 1317, 1426, 1258, 1407, 1301, 1152, 1219, 1327, 1112, 1232, 1161, 1168, 1136, 1185, 1246, 1311, 1391, 1201, 1098, 1096, 1102, 1011, 1060, 1021, 1018, 1179, 1011, 1005, 826, 910, 1088,
     919, 946, 932, 1174, 887, 1048, 953, 929, 759, 805, 704, 699, 854, 783, 809, 727, 815, 676, 795, 803, 661, 638, 803, 696, 695, 688, 612, 683, 764, 614, 625, 656, 757, 667, 604, 541, 717, 582, 737, 606, 584, 553, 576, 609, 590, 735, 721, 659, 714, 574, 714, 678, 507, 552, 532, 594, 536, 561, 532, 534, 456, 523, 537, 554, 636, 557, 534, 526, 573, 514, 441, 478, 331, 480, 499, 421, 396, 471, 350, 419, 410, 357, 329, 395, 366, 377, 383, 371, 343, 474, 444, 312, 348, 216, 311, 325, 431, 330, 289, 355, 301, 365, 283, 330, 242, 266, 313, 366, 268, 284, 231, 239, 308, 187, 308, 291, 244, 320, 262, 261, 203, 217, 260, 259, 226, 186, 147, 241, 240, 164, 237, 226, 182, 113, 214, 265, 240, 215, 167, 133, 184, 146, 277, 198, 182, 201, 142, 189, 196, 147, 219, 217, 228, 186, 264, 212, 258, 267, 181, 141, 113, 168, 190, 184, 147, 185, 209, 188, 181, 135, 191, 113, 162, 128, 195, 217, 196, 249, 172, 213, 182, 168, 188, 141, 135, 149, 141, 79, 128, 148, 178, 200, 132, 166, 126, 156, 208, 140, 133, 121, 104, 140, 103, 110, 123, 109, 92, 105, 102, 111, 77, 89, 118, 124, 125, 104, 107, 122, 60, 77, 43, 99, 45, 112, 57, 51, 80, 44, 52, 54, 85, 85, 69, 39, 58, 46, 35, 103, 50, 56, 93, 85, 56, 55, 37, 43, 73, 46, 92, 69, 63, 79, 91, 23, 56, 62, 54, 44, 59, 57, 37, 24, 29, 57, 42, 51, 60, 13, 34, 33, 71, 20, 33, 57, 31, 40, 101, 36, 97, 58, 59, 56, 50, 36, 30, 5, 12, 19, 26, 39, 25, 61, 76, 48, 77, 29, 44, 5, 25, 12, 7, 18, 42, 17, 57, 20, 40, 27, 26, 17, 26, 38, 60, 38, 42, 40, 64, 23, 22, 17, 42, 34, 29, 30, 19, 30, 21, 7, 6, 3, 15, 0, 13, 1, 4, 32, 12, 30, 16, 17, 18, 11, 21, 10, 4, 8, 4, 4, 5, 20, 12, 14, 9, 7, 9, 7, 21, 4, 6, 7, 14, 11, 12, 8, 4, 11, 10, 12, 15, 10, 35, 13, 7, 7, 7, 7, 7, 4, 6, 5, 24, 8, 5, 2, 10, 19, 4, 5, 1, 18, 28, 26, 26, 39, 18, 7, 13, 0, 1, 15, 13, 8, 33, 4, 7, 2, 12, 4, 0, 2, 0, 0, 0, 4, 28, 37, 3, 10, 7, 19, 12, 7, 4, 3, 16, 13, 10, 6, 12, 12, 5, 4, 13, 9, 12, 12, 9, 12, 12, 13, 14, 9, 15, 14, 3, 2, 5, 6, 0, 9, 3, 6, 8, 5, 4, 6, 2, 5, 3, 2, 4, 1, 3, 5, 2, 5, 3, 5, 4, 3, 1, 2, 3, 1, 1, 0, 2, 0, 0, 0, 0, 1, 1, 3, 4, 2, 7, 8, 2, 6, 6, 9, 10, 9, 9, 9, 9, 2, 1, 2, 0, 0, 1, 1, 0, 0, 1, 1, 5, 2, 3, 4, 3, 1, 0, 0, 2, 0, 2, 3, 1, 22, 41, 19, 1, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 1, 1, 3, 0, 4, 5, 7, 9, 12, 15, 15, 10, 7, 22, 22, 15, 24, 19, 16, 12, 0]

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

font_legend = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

y = y[100:300]
max_size = 4
# x = [[i/len(y) * max_size] for i in range(len(y))]
data = [[2 + i/len(y) * max_size, y[i]] for i in range(len(y))]


# data = np.genfromtxt('data.txt')

data = np.array(data)

# part of code from:
# https://stackoverflow.com/questions/26936094/python-load-data-and-do-multi-gaussian-fit

def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset


def four_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, h4, c4, w4, offset):
    return np.maximum(gaussian(x, h1, c1, w1, offset=0) +
            gaussian(x, h2, c2, w2, offset=0) +
            gaussian(x, h4, c4, w4, offset=0) +
            gaussian(x, h3, c3, w3, offset=0) , 0)
            # gaussian(x, h3, c3, w3, offset=0) + offset, 0)

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
            gaussian(x, h2, c2, w2, offset=0) +
            gaussian(x, h3, c3, w3, offset=0) + offset)

def two_gaussians(x, h1, c1, w1, h2, c2, w2, offset):
    return three_gaussians(x, h1, c1, w1, h2, c2, w2, 0, 0, 1, offset)

# errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)**2
def errfunc3(p, x, y): return (four_gaussians(x, *p) - y)**2
def errfunc2(p, x, y): return (two_gaussians(x, *p) - y)**2


guess3 = [2.0e+03, 2.5, 0.005,
          2.0e+03, 5.0, 0.005,
          2.0e+03, 4.0, 0.005,
          2.0e+03, 3.0, 0.005,
          0]

# guess3 = [1000, 2, 0.01,     1000, 5, 0.01,     1000, 4, 0.01,  1000, 3, 0.01 ,    0]
# I removed the peak I'm not too sure about
guess2 = [1000, 2, 0.01,     1000, 5, 0.01,     0]

optim3, success = optimize.leastsq(
    errfunc3, guess3[:], args=(data[:, 0], data[:, 1]))
optim2, success = optimize.leastsq(
    errfunc2, guess2[:], args=(data[:, 0], data[:, 1]))

print(optim3, success)


fig, ax = plt.subplots()

ax.plot(data[:, 0], data[:, 1], lw=5, c='g', label='origin data')
ax.plot(data[:, 0], four_gaussians(data[:, 0], *optim3),
         lw=3, c='b', label='fit of 4 Gaussians')

ax.plot(data[:, 0], gaussian(data[:, 0], optim3[0], optim3[1], optim3[2], optim3[-1]),
         lw=1)
ax.plot(data[:, 0], gaussian(data[:, 0], optim3[3], optim3[4], optim3[5], optim3[-1]),
         lw=1)
ax.plot(data[:, 0], gaussian(data[:, 0], optim3[6], optim3[7], optim3[8], optim3[-1]),
         lw=1)
ax.plot(data[:, 0], gaussian(data[:, 0], optim3[9], optim3[10], optim3[11], optim3[-1]),
         lw=1)

ax.tick_params(labelsize=16)
ax.set_xlabel(r'$D_{O-O}, \AA$', font_axis)

ax.set_ylabel(r'Count', font_axis)
# plt.plot(data[:, 0], two_gaussians(data[:, 0], *optim2),
#          lw=1, c='r', ls='--', label='fit of 2 Gaussians')
ax.grid()

plt.legend(loc='best')
plt.show()
# plt.savefig('result.png')
