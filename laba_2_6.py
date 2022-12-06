import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def reverse_log(x, eps=1.0e-05):
    return -math.log( -x + eps )

# создаём точки для проверки
p_num = 101
As = np.logspace(-2, 5, p_num)
# массив для результатов (ожидаем 2 полюса, т. к. характеристическое уравнение системы имеет порядок 2)
Ps_positive_a = np.empty((p_num, 2), dtype=complex)
Ps_negative_a = np.empty((p_num, 2), dtype=complex)
b = 2
c = 10
'''
# Щупаем систему положительными значениями параметра a

for i in range(p_num):
    # рассчитываем нули, полюсы и усиление для проверяемой системы
    z,p,k = signal.tf2zpk([c, 0], [b, As[i], 1])
    # выводим полюсы
    if i % 10 == 0:
        print('a={:.6f}\tp_1=[{:.6f},\t{:.6f}]\tp_2=[{:.6f},\t{:.6f}]'.format(As[i],
                                                                              np.real(p[0]), 
                                                                              np.imag(p[0]),
                                                                              np.real(p[1]),
                                                                              np.imag(p[1])))
    
    Ps_positive_a[i][0] = p[0]
    Ps_positive_a[i][1] = p[1]
plt.figure(figsize=(9,9))
plt.plot([np.real(Ps_positive_a[i][0]) for i in range(1, p_num)],
         [np.imag(Ps_positive_a[i][0]) for i in range(1, p_num)], '.-',
         [np.real(Ps_positive_a[i][1]) for i in range(1, p_num)],
         [np.imag(Ps_positive_a[i][1]) for i in range(1, p_num)], '.-')
plt.title("Poles for positive 'a' (stable)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.figure(figsize=(9,9))
plt.plot([reverse_log(np.real(Ps_positive_a[i][0])) for i in range(1, p_num)],
         [            np.imag(Ps_positive_a[i][0])  for i in range(1, p_num)], '.-',
         [reverse_log(np.real(Ps_positive_a[i][1])) for i in range(1, p_num)],
         [            np.imag(Ps_positive_a[i][1])  for i in range(1, p_num)], '.-')
plt.title("Poles for positive 'a' (stable)")
plt.xlabel("-log(-Re)")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.show()

# Щупаем систему отрицательными значениями параметра a

for i in range(p_num):
    # рассчитываем нули, полюсы и усиление для проверяемой системы
    z,p,k = signal.tf2zpk([c, 0], [b, -As[i], 1])
    # выводим полюсы
    if i % 10 == 0:
        print('a={:.6f}\tp_1=[{:.6f},\t{:.6f}]\tp_2=[{:.6f},\t{:.6f}]'.format(-As[i],
                                                                              np.real(p[0]), 
                                                                              np.imag(p[0]),
                                                                              np.real(p[1]),
                                                                              np.imag(p[1])))
    
    Ps_negative_a[i][0] = p[0]
    Ps_negative_a[i][1] = p[1]
plt.figure(figsize=(9,9))
plt.plot([np.real(Ps_negative_a[i][0]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][0]) for i in range(1, p_num)], '.-',
         [np.real(Ps_negative_a[i][1]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][1]) for i in range(1, p_num)], '.-')
plt.title("Poles for negative 'a' (unstable)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.figure(figsize=(9,9))
plt.xscale("log")
plt.plot([np.real(Ps_negative_a[i][0]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][0]) for i in range(1, p_num)], '.-',
         [np.real(Ps_negative_a[i][1]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][1]) for i in range(1, p_num)], '.-')
plt.title("Poles for negative 'a' (unstable)")
plt.xlabel("log(Re)")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.show()


# Подадим на вход устойчивой системы ступеньку

a_stable = 3
model = signal.lti([c, 0], [b, a_stable, 1])
T = np.linspace(0, 20, 1001)
S = [1 for t in T]
Tout, yout, xout = signal.lsim(model, S, T)
plt.figure()
plt.yscale('log')
plt.plot(Tout, yout)
plt.title("Response of a stable system to a single step input signal")

# Построим АЧХ и ФЧХ устойчивой системы

Omegas, A, Phi = signal.bode(model)
fs = [0.5*w/math.pi for w in Omegas]
plt.figure()
plt.semilogx(fs, A)
plt.title("Amplitude-frequency responce of a stable system")
plt.figure()
plt.semilogx(fs, Phi)
plt.title("Phase-frequency responce of a stable system")
plt.show()


# Подадим на вход неустойчивой системы ступеньку

a_unstable = -3
model = signal.lti([c, 0], [b, a_unstable, 1])
Tout, yout, xout = signal.lsim(model, S, T)
plt.figure()
plt.yscale('log')
plt.plot(Tout, yout, "r-")
plt.title("Response of an unstable system to a single step input signal")

# Построим АЧХ и ФЧХ устойчивой системы

Omegas, A, Phi = signal.bode(model)
fs = [0.5*w/math.pi for w in Omegas]
plt.figure()
plt.semilogx(fs, A, "r-")
plt.title("Amplitude-frequency responce of an unstable system")
plt.figure()
plt.semilogx(fs, Phi, "r-")
plt.title("Phase-frequency responce of an unstable system")
plt.show()
'''
'''
# Исследуем, как меняются положения полюсов замкнутой системы при изменении параметра a
# Щупаем систему положительными значениями параметра a

for i in range(p_num):
    # рассчитываем нули, полюсы и усиление для проверяемой системы
    z,p,k = signal.tf2zpk([c, 0], [b, As[i], 1])
    # выводим полюсы
    if i % 10 == 0:
        print('a={:.6f}\tp_1=[{:.6f},\t{:.6f}]\tp_2=[{:.6f},\t{:.6f}]'.format(As[i] - 10,
                                                                              np.real(p[0]), 
                                                                              np.imag(p[0]),
                                                                              np.real(p[1]),
                                                                              np.imag(p[1])))
    
    Ps_positive_a[i][0] = p[0]
    Ps_positive_a[i][1] = p[1]
plt.figure(figsize=(9,9))
plt.plot([np.real(Ps_positive_a[i][0]) for i in range(1, p_num)],
         [np.imag(Ps_positive_a[i][0]) for i in range(1, p_num)], 'g.-',
         [np.real(Ps_positive_a[i][1]) for i in range(1, p_num)],
         [np.imag(Ps_positive_a[i][1]) for i in range(1, p_num)], 'c.-')
plt.title("Poles for a stable loop system")
plt.xlabel("Re")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.figure(figsize=(9,9))
plt.plot([reverse_log(np.real(Ps_positive_a[i][0])) for i in range(1, p_num)],
         [            np.imag(Ps_positive_a[i][0])  for i in range(1, p_num)], 'g.-',
         [reverse_log(np.real(Ps_positive_a[i][1])) for i in range(1, p_num)],
         [            np.imag(Ps_positive_a[i][1])  for i in range(1, p_num)], 'c.-')
plt.title("Poles for a stable loop system")
plt.xlabel("-log(-Re)")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.show()

# Щупаем систему отрицательными значениями параметра a

for i in range(p_num):
    # рассчитываем нули, полюсы и усиление для проверяемой системы
    z,p,k = signal.tf2zpk([c, 0], [b, -As[i], 1])
    # выводим полюсы
    if i % 10 == 0:
        print('a={:.6f}\tp_1=[{:.6f},\t{:.6f}]\tp_2=[{:.6f},\t{:.6f}]'.format(-As[i] - 10,
                                                                              np.real(p[0]), 
                                                                              np.imag(p[0]),
                                                                              np.real(p[1]),
                                                                              np.imag(p[1])))
    
    Ps_negative_a[i][0] = p[0]
    Ps_negative_a[i][1] = p[1]
plt.figure(figsize=(9,9))
plt.plot([np.real(Ps_negative_a[i][0]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][0]) for i in range(1, p_num)], 'g.-',
         [np.real(Ps_negative_a[i][1]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][1]) for i in range(1, p_num)], 'c.-')
plt.title("Poles for an unstable loop system")
plt.xlabel("Re")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.figure(figsize=(9,9))
plt.xscale("log")
plt.plot([np.real(Ps_negative_a[i][0]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][0]) for i in range(1, p_num)], 'g.-',
         [np.real(Ps_negative_a[i][1]) for i in range(1, p_num)],
         [np.imag(Ps_negative_a[i][1]) for i in range(1, p_num)], 'c.-')
plt.title("Poles for an unstable loop system")
plt.xlabel("log(Re)")
plt.ylabel("Im")
plt.grid("on")
plt.legend(["pole 1", "pole 2"], loc="best")
plt.show()
'''
# Отследим реакцию замкнутой системы на ступенчатый сигнал

# Подадим на вход (ранее) устойчивой системы ступеньку

a_stable = 3
model = signal.lti([c, 0], [b, a_stable + 10, 1])
T = np.linspace(0, 20, 1001)
S = [1 for t in T]
Tout, yout, xout = signal.lsim(model, S, T)
plt.figure()
plt.yscale('log')
plt.plot(Tout, yout)
plt.title("Response to a single step input signal (a = 3)")

# Построим АЧХ и ФЧХ (ранее) устойчивой системы

Omegas, A, Phi = signal.bode(model)
fs = [0.5*w/math.pi for w in Omegas]
plt.figure()
plt.semilogx(fs, A)
plt.title("Amplitude-frequency responce (a = 3)")
plt.figure()
plt.semilogx(fs, Phi)
plt.title("Phase-frequency responce (a = 3)")
plt.show()


# Подадим на вход (ранее) неустойчивой системы ступеньку

a_unstable = -3
model = signal.lti([c, 0], [b, a_unstable + 10, 1])
Tout, yout, xout = signal.lsim(model, S, T)
plt.figure()
plt.yscale('log')
plt.plot(Tout, yout, "r-")
plt.title("Response to a single step input signal (a = -3)")

# Построим АЧХ и ФЧХ (ранее) неустойчивой системы

Omegas, A, Phi = signal.bode(model)
fs = [0.5*w/math.pi for w in Omegas]
plt.figure()
plt.semilogx(fs, A, "r-")
plt.title("Amplitude-frequency responce (a = -3)")
plt.figure()
plt.semilogx(fs, Phi, "r-")
plt.title("Phase-frequency responce (a = -3)")
plt.show()
