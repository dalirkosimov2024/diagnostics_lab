import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plot = "no"

path="/home/dalir/Documents/cdt/diagnostics_lab/mcf/thompson_scattering/docs/data"

intensity_data = np.loadtxt(f"{path}/intensity.dat")
lambda_data = np.loadtxt(f"{path}/lambda.dat")

if plot == "yes":
    plt.imshow(intensity_data)
    plt.show()



    plt.scatter(lambda_data[142]/10, intensity_data[142], s=10)
    points_to_mask = [649, 667, 684, 706]
    for point in points_to_mask:
        plt.axvline(point, color="red", linestyle="--")
    plt.show()

def mask(lambda_data, intensity_data):

    intensity_data = intensity_data[142]
    lambda_data = lambda_data[142]/10
    for value in lambda_data:
        if value > 649:
            if value < 669:
                print(f"value: {value}\n\n")
                position = np.where(lambda_data == value)
                lambda_data = np.delete(lambda_data, position)
                intensity_data = np.delete(intensity_data, position)

    for value in lambda_data:
        if value > 678:
            if value < 707:
                print(f"value: {value}\n\n")
                position = np.where(lambda_data == value)
                lambda_data = np.delete(lambda_data, position)
                intensity_data = np.delete(intensity_data, position)
    return lambda_data, intensity_data


def fit():
    def gaussian(lambda_s, *params):
        A_0 = params[0] # intensity
        lambda_i = params[1] # incident photons
        sigma_lambda = params[2] # width (nm)

        return A_0 * np.exp(-0.5* ((lambda_s - lambda_i)/sigma_lambda)**2)

    guess = [1880, 694, 70]

    popt, pocv = curve_fit(gaussian,lambda_data, intensity_data, p0=guess)
    gaussian_fit = gaussian(lambda_data, *popt)
    return gaussian_fit

def plot():
    plt.scatter(lambda_data, intensity_data, s=10)
    plt.plot(lambda_data,gaussian_fit)
    plt.show()


if __name__ == "__main__":
    lambda_data, intensity_data = mask(lambda_data, intensity_data)
    gaussian_fit = fit()
    plot()



    






