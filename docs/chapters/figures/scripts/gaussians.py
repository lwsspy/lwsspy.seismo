import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq

# Internal
import lwsspy.base as lbase
import lwsspy.plot as lplt
import lwsspy.seismo as lseis
lplt.updaterc()

# Setup time vector
t = np.arange(0, 1.0, 0.00001)  # Longer than plot for resolution in F domain!
Nt = len(t)
maxnt = Nt//2
dt = np.diff(t)[0]
t0 = 0.05  # Time shift
f0 = 20.0  # Dominant frequency

# Setup Frequency Vector
freq = fftfreq(Nt, dt)[:maxnt]

# Compute Gaussians
g = lseis.gaussiant(t, t0=t0, f0=f0)
dg = lseis.dgaussiant(t, t0=t0, f0=f0)

# Compute Spectrums
fg = np.abs(fft(g))[:maxnt]
fdg = np.abs(fft(dg))[:maxnt]


# Create figure
plt.figure()
plt.subplots_adjust(hspace=0.35)
plt.subplot(211)
plt.plot(t, g, label='Gaussian')
plt.plot(t, dg, label='d/dt(Gaussian)')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.xlim(0.0, 0.2)
plt.legend(ncol=2, frameon=False)

plt.subplot(212)
plt.plot(freq, fg/np.max(fg), label='Gaussian')
plt.plot(freq, fdg/np.max(fdg), label='d/dt(Gaussian)')
plt.legend(ncol=2, frameon=False)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Norm. Amplitude")
plt.xlim(0.0, 100.0)

outnamesvg = os.path.join(lbase.DOCFIGURES, "gaussians.svg")
outnamepdf = os.path.join(lbase.DOCFIGURES, "gaussians.pdf")

plt.savefig(outnamesvg)
plt.savefig(outnamepdf)
