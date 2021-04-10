from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 200
T = 1.0 / 10.0

x = np.linspace(0.0, N * T, N)
y_1 = 10 * np.sin(2 * 2.0 * np.pi * x)
y_2 = 10 * np.sin(4 * 2.0 * np.pi * x)

y_3 = 10 * np.cos(2 * 2.0 * np.pi * x)
y_4 = 10 * np.cos(4 * 2.0 * np.pi * x)

yf_1 = fft(y_1)
yf_2 = fft(y_2)
# yf_3 = fft(y_3)
# yf_4 = fft(y_4)

xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
mag_1 = 2.0 / N * np.abs(yf_1[0:N // 2])
mag_2 = 2.0 / N * np.abs(yf_2[0:N // 2])
# mag_3 = 2.0 / N * np.abs(yf_3[0:N // 2])
# mag_4 = 2.0 / N * np.abs(yf_4[0:N // 2])

# save data
values_1 = {'Time': xf[1:], 'Mag_1': mag_1[1:], 'Mag_2': mag_2[1:]}
df_w_1 = pd.DataFrame(values_1, columns=['Time', 'Mag_1', 'Mag_2'])
df_w_1.to_csv("data/fft-data-2.csv", index=False, header=True)

plt.subplot(2, 1, 1)
plt.plot(x, y_1)
plt.plot(x, y_2)
# plt.plot(x, y_3)
# plt.plot(x, y_4)
plt.subplot(2, 1, 2)
plt.semilogy(xf[1:], mag_1[1:])  # sine
plt.semilogy(xf[1:], mag_2[1:])
# plt.semilogy(xf, mag_3)  # cos
# plt.semilogy(xf, mag_4)
plt.grid()
plt.show()
