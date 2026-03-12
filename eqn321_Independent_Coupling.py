from numba import jit, njit, prange
import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
import numba
#numba.config.NUMBA_DEFAULT_NUM_THREADS = 4

def ExpDecayRatefun(A):
    T_step = 0.05
    B = np.sort(A)
    n = len(B)
    T = T_step
    k = 0
    countNum = 0
    dist_list = []
    Tspan_list = []
    while k <= n-1:
        if B[k] <= T:
            countNum = countNum + 1
            k = k + 1
        else:
            dist_list.append(countNum)
            Tspan_list.append(T)
            T = T + T_step
    l = len(Tspan_list)
    dist = np.array(dist_list)
    Tspan = np.array(Tspan_list)
    dist = np.log(np.ones(l) - dist / n)
    end_l = round(l*4/5)
    slope = np.polyfit(Tspan[:end_l], dist[:end_l], 1)
    return Tspan, dist, slope



@njit
def eqn321_Independent_Coupling():
    def eqn321Fun(xx):
        yy = - pow(xx, 3)
        return yy
    couplingtime = 0

    sample1 = low + h / 2.0 + round((low + Sp * random.random() - low - h / 2.0) / h) * h
    sample2 = low + h / 2.0 + round((low + Sp * random.random() - low - h / 2.0) / h) * h

    sample1_n = round((sample1 - low - h / 2.0) / h)
    sample2_n = round((sample2 - low - h / 2.0) / h)

    sample1_count = 0
    sample2_count = 0
    sample1_time = 0
    sample2_time = 0

    tmp = eqn321Fun(sample1)
    M = 0.5 * max(eps0 * eps0 - abs(tmp) * h, 0.0)
    q1 = np.zeros(2)
    q1[0] = max(tmp, 0.0) / h + M / (h * h)
    q1[1] = -min(tmp, 0.0) / h + M / (h * h)
    Lambda =q1[0]+q1[1]
    rnd1 = random.random()
    tau = -math.log(1 - rnd1) / Lambda
    sample1_time += tau

    tmp = eqn321Fun(sample2)
    M = 0.5 * max(eps0 * eps0 - abs(tmp) * h, 0.0)
    q2 = np.zeros(2)
    q2[0] = max(tmp, 0.0) / h+M / (h * h)
    q2[1] = -min(tmp, 0.0) / h+M / (h * h)
    Lambda =q2[0]+q2[1]
    rnd1 = random.random()
    tau = -math.log(1-rnd1) / Lambda
    sample2_time += tau

    while sample1_time < end_time:
    # step of sample1
        if sample1_time <= sample2_time:
            rnd2 = random.random()
            mu_number = 0
            amu = q1[mu_number]
            while amu < rnd2 * Lambda:
                mu_number += 1
                amu += q1[mu_number]

            if mu_number == 0:
                sample1 = sample1 + h
            elif mu_number == 1:
                sample1 = sample1 - h
            else:
                print("nu_number is wrong")

            sample1_n = round((sample1 - low - h / 2.0) / h)

            if sample1_n == sample2_n:
                couplingtime = sample1_time
                break
            tmp = eqn321Fun(sample1)
            M = 0.5 * max(eps0 * eps0 - abs(tmp) * h, 0.0)
            q1[0] = max(tmp, 0.0) / h + M / (h * h)
            q1[1] = -min(tmp, 0.0) / h + M / (h * h)
            Lambda = q1[0] + q1[1]
            rnd1 = random.random()
            tau = -math.log(1 - rnd1) / Lambda
            sample1_time += tau
        else:
            rnd2 = random.random()
            mu_number = 0
            amu = q2[mu_number]
            while amu < rnd2 * Lambda:
                mu_number += 1
                amu += q2[mu_number]

            if mu_number == 0:
                sample2 = sample2 + h
            elif mu_number == 1:
                sample2 = sample2 - h
            else:
                print("nu_number is wrong")

            sample2_n = round((sample2 - low - h / 2.0) / h)

            if sample1_n == sample2_n:
                couplingtime = sample2_time
                break
            tmp = eqn321Fun(sample2)
            M = 0.5 * max(eps0 * eps0 - abs(tmp) * h, 0.0)
            q2[0] = max(tmp, 0.0) / h + M / (h * h)
            q2[1] = -min(tmp, 0.0) / h + M / (h * h)
            Lambda = q2[0] + q2[1]
            rnd1 = random.random()
            tau = -math.log(1 - rnd1) / Lambda
            sample2_time += tau

    return couplingtime

@njit
#@njit(parallel=True)
def my_fun(loop):
    data = np.zeros(loop)
    for i in prange(loop):
        data[i] = eqn321_Independent_Coupling()
        print("Loop is", i)
    return data
##########################################################################################

start = time.time()

N = 240
loop = 50000
end_time = 200000

eps0 = math.sqrt(2)
eps = 0.5 * eps0 * eps0

low = -3.0
Sp = 6
h = Sp / N

data = my_fun(loop)
#print(data)

Tspan, dist, slope = ExpDecayRatefun(data)

end = time.time()
print("Total loop is ", loop)
print("Total time is = %s" % (end - start))

plt.plot(Tspan, dist, 'b')
plt.plot(Tspan, Tspan * slope[0] + slope[1], 'r--')
plt.title('N = %s, slope = %s' % (str(N), str(slope[0])))
plt.xlim([0, 8])
plt.ylim([-10, 0])
plt.legend(["the coupling time distribution", "the linear fitting"])
plt.show()