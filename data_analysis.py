import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def calculateMass(eta, phi, pt):
    mass_sqrd = 2 * abs(pt[0]) * abs(pt[1]) * (math.cosh(eta[0] - eta[1]) - math.cos(phi[0] - phi[1]))
    return math.sqrt(mass_sqrd)


def calculateMassCombinations(eta, phi, pt):
    indexes = combinations(range(len(eta)), 2)
    masses = []

    for index in indexes:
        eta_pair = [eta[index[0]], eta[index[1]]]
        phi_pair = [phi[index[0]], phi[index[1]]]
        pt_pair = [pt[index[0]], pt[index[1]]]

        masses.append(round(calculateMass(eta_pair, phi_pair, pt_pair), 2))
    return masses

def binEvents(bin_start, bin_end, bin_size, file_name):
    with open(f'Diphotons{file_name}.dat', 'r') as file:
        bin_number = (bin_end - bin_start) // bin_size
        bin = [0 for i in range(bin_number)]

        eta = []
        phi = []
        pt = []

        for line in file:
            if '#' in line:
                continue

            data = line.split()
            if data == []:
                continue

            if data[0] != '0' and data[1] == '0':
                eta.append(float(data[2]))
                phi.append(float(data[3]))
                pt.append(float(data[4]))
            elif data[0] == '0':
                masses = calculateMassCombinations(eta, phi, pt)
                
                for mass in masses:
                    mass = int(mass)

                    index = (mass - bin_start) // bin_size

                    if index >= 0 and index < (bin_end - bin_start) // bin_size:
                        bin[index] += 1


                eta = []
                phi = []
                pt = []
        return bin


def main():

    bin_start = 100
    bin_end = 300
    bin_size = 2
    bin_number = (bin_end - bin_start) // bin_size

    file_name = input("File name:")

    
    bin_x = [i * bin_size + bin_start for i in range(1, bin_number + 1)]
    bin_y = binEvents(bin_start, bin_end, bin_size, file_name)

    bcoefficients = np.polyfit(bin_x, bin_y, 5)
    sbcoefficients = np.polyfit(bin_x, bin_y, 50)
    
    bkg_fit = np.poly1d(bcoefficients)
    sb_fit = np.poly1d(sbcoefficients)
    fit_x = np.linspace(bin_start, bin_end, bin_number*3)
    bfit_y = []
    sbfit_y = []
    ssfit_y = []

    for x in fit_x:
        bfit_y.append(bkg_fit(x))
        sbfit_y.append(sb_fit(x))
        ssfit_y.append(sb_fit(x) - bkg_fit(x))

    signal_x = bin_x
    signal_y = []
    significant = 0
    significant_x = 0
    significant_y = 0

    for i in range(len(bin_x)):
        signal = bin_y[i] - bkg_fit(bin_x[i])
        signal_y.append(signal)
        significance = signal / math.sqrt(bkg_fit(bin_x[i]))
        if significance > significant and bin_y[i] > 2000:
            significant = significance
            significant_x = bin_x[i]
            significant_y = bin_y[i]
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, height_ratios=[2, 1])
    fig.subplots_adjust(hspace=0)
    ax = plt.gca()
    ax.set_xlim([significant_x - 20, significant_x + 40])

    ax1.set_title('Higgs discovery through the diphoton channel')
    ax1.set_ylabel('Events / 2 GeV')
    ax1.plot(fit_x, bfit_y, 'r--', label='Bkg fit(4$^{\\text{th}}$order polynomial)')
    ax1.plot(fit_x, sbfit_y, 'r-', label='Sig + Bkg fit')
    ax1.plot(bin_x, bin_y, 'ko', label='Data')
    ax1.axvline(significant_x, c='b')
    ax1.text(significant_x + 2, significant_y + 500, f"$\sigma = {significant:.2f}$")
    ax1.legend()

    ax2.set_ylabel('Events - Background')
    ax2.set_xlabel('m$_{\gamma \gamma}$ [GeV]')
    ax2.plot(fit_x, ssfit_y, 'r-')
    ax2.axhline(c='r', ls='--')
    ax2.axvline(significant_x, c='b')
    ax2.plot(signal_x, signal_y, 'ko')

    plt.show()


if __name__ == '__main__':
    main()
