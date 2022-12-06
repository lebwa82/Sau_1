import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#4. Создайте модель в scipy.signal, описывающую систему. Значения констант следует для простоты принять равными небольшим целым числам.
a = 3
b = 2
c = 10
model = signal.lti([1], [a, 1])  # returns TransferFunctionContinuous (subclass of lti) object


#5. Подавая на вход модели синусоидальный сигнал нескольких различных амплитуд, но с постоянной частотой, отследите реакцию модели (амплитуду и фазу выходного сигнала). Какой вывод можно сделать?

freq = 0.1
Amps = [0.003, 0.1, 3.0, 100.0]
T = np.linspace(0, 200.0/freq, 10001)

for amp in Amps:
    S = [amp * math.sin(t*2*math.pi*freq) for t in T]
    Tout, yout, xout = signal.lsim(model, S, T)
    plt.figure()
    plt.plot(T[-200:-1], S[-200:-1])
    plt.title("Input signal with amplitude {}".format(amp))
    plt.figure()
    plt.plot(Tout[-200:-1], yout[-200:-1])
    plt.title("Output signal")
    plt.show()
    print("Input amlplitude: {};\toutput amplitude: {}".format(amp, (max(yout[-200:-1])-min(yout[-200:-1]))/2))
    print("oriental point: {}\t{}".format(Tout[-100], yout[-100]))

#6. Подавая на вход модели синусоидальный сигнал нескольких различных частот, но с постоянной амплитудой, отследите реакцию модели (амплитуду и фазу выходного сигнала). Какой вывод можно сделать?
Frs = [0.01, 0.045, 0.21, 1.0]
Amps = [1, 1, 1, 1]
for i in range(0, 4):
    T = np.linspace(0, 200.0/Frs[i], 10001)
    S = [math.sin(t*2*math.pi*Frs[i]) for t in T]
    Tout, yout, xout = signal.lsim(model, S, T)
    '''
    plt.figure()
    plt.plot(T[-200:-1], S[-200:-1])
    plt.title("Input signal with frequency {}".format(Frs[i]))
    plt.figure()
    plt.plot(Tout[-200:-1], yout[-200:-1])
    plt.title("Output signal")
    plt.show()
    '''
    Amps[i] = (max(yout[-200:-1])-min(yout[-200:-1]))/2
    print("Input frequency: {};\toutput amplitude: {}".format(Frs[i], Amps[i]))
    print("oriental point: {}\t{}".format(Tout[-100], yout[-100]))

    
#7. Постройте графики частотных характеристик моделируемой системы. Оцените степень их совпадения с рассчитанным теоретически.
Omegas, A, Phi = signal.bode(model) # returns magnitude and phase arrays for a "reasonable" set of frequencies chosen by function
fs = [0.5*w/math.pi for w in Omegas] 

A_calc = [20*math.log10(c / (math.sqrt((1-b*w*w)**2+a*a*w*w))) for w in Omegas]
#plt.figure()
#plt.semilogx(fs, A)
#plt.title("Amplitude-frequency responce (signal.bode opinion)")
#plt.figure()
#plt.semilogx(fs, A_calc)
#plt.title("Amplitude-frequency responce (calculated)")
#plt.show()

Phi_calc = [- 180 / math.pi * math.atan2(a*w, 1-b*w*w) for w in Omegas]
#plt.figure()
#plt.semilogx(fs, Phi)
#plt.title("Phase-frequency responce (signal.bode opinion)")
#plt.figure()
#plt.semilogx(fs, Phi_calc)
#plt.title("Phase-frequency responce (calculated)")
#plt.show()
'''
#8. Сравните вычисленные и экспериментальные АЧХ и ФЧХ.
Frs = [0.01, 0.045, 0.21, 1.0] #они не поменялись, я просто напоминаю
Phases = [-0.2, -0.79, -2.13, -2.9]

plt.figure()
plt.semilogx(fs, A_calc)
plt.semilogx(Frs, [(20 * math.log10(am)) for am in Amps], 'o')
plt.title("Amplitude-frequency responce (comparison of calculated and experimental)")
plt.figure()
plt.semilogx(fs, Phi_calc)
plt.semilogx(Frs, [(ph * 180 / math.pi) for ph in Phases], 'o')
plt.title("Phase-frequency responce (comparison of calculated and experimental)")
plt.show()



#10. Подайте на вход модели единичное ступенчатое воздействие. Как соотносятся реакция системы на него и переходная функция?
S = [1 for t in T[:2001]]
H = [(10.0 + 10 * math.exp(-t) - 20 * math.exp(-t/2)) for t in T[:2001]]
Tout, yout, xout = signal.lsim(model, S, T[:2001])
plt.figure()
plt.plot(Tout[:2001], yout[:2001])
plt.title("Output signal")
plt.figure()
plt.plot(T[:2001], H)
plt.title("Transient function")
plt.show()


'''
