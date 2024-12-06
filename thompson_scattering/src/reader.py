import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.lines as mlines

plot = "no"
looper="yes"

path="/home/dalir/Documents/cdt/diagnostics_lab/mcf/diagnostics_lab/thompson_scattering/docs/data"
# I = intensity, L = lambda, A = anlge
I= np.loadtxt(f"{path}/intensity.dat")
L = np.loadtxt(f"{path}/lambda.dat")/10
A = np.loadtxt(f"{path}/angle.dat")
R = np.loadtxt(f"{path}/radius.dat")


if plot == "yes":
    plt.imshow(I)
    plt.xlabel("Wavelength (a.u.)")
    plt.ylabel("Spatial position (a.u.)")
    plt.show()



    plt.scatter(L[i], I[i], s=10)
    points_to_mask = [649, 667, 684, 706]
    for point in points_to_mask:
        plt.axvline(point, color="red", linestyle="--")
    cropped_area = mlines.Line2D([],[], linestyle="--", color="red", label = "Areas to mask")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend(handles = [cropped_area])

    
    plt.show()

def mask(i,L,I):

    intensity_data = I[i]
    lambda_data = L[i]
    for value in lambda_data:
        if value > 649:
            if value < 669:
                position = np.where(lambda_data == value)
                lambda_data = np.delete(lambda_data, position)
                intensity_data = np.delete(intensity_data, position)

    for value in lambda_data:
        if value > 678:
            if value < 707:
                position = np.where(lambda_data == value)
                lambda_data = np.delete(lambda_data, position)
                intensity_data = np.delete(intensity_data, position)
    return lambda_data, intensity_data


def fit(i):
    def gaussian(lambda_s, *params):
        A_0 = params[0] # intensity
        lambda_i = params[1] # incident photons
        sigma_lambda = params[2] # width (nm)

        return A_0 * np.exp(-0.5* ((lambda_s - lambda_i)/sigma_lambda)**2)

    guess = [1880, 694, 35]

    popt, pcov = curve_fit(gaussian,lambda_data, intensity_data, p0=guess)
    gaussian_fit = gaussian(L[i], *popt)

    error = np.sqrt(np.diag(pcov))

    return gaussian_fit, popt, error[2]

def electron_temp_calc(i):
    theta = A[i]
    sigma_lambda = popt[2]
    lambda_i = 694.3
    T_e = (sigma_lambda / (lambda_i * np.sin(theta/2)))**2 * 1.24*10**5

    new_error = 2 * error
    return T_e, new_error


def plot_single(i):
    plt.scatter(lambda_data, intensity_data, s=10)
    plt.plot(L[i],gaussian_fit,color="orange")

    gaussian_line = mlines.Line2D([],[],color="orange", label="Gaussian fit")
    info = mlines.Line2D([],[], linestyle="none", label = r"$\sigma_\lambda$ = "f"{round(popt[2],2)}"r" $\pm$ "f"{round(error,2)} nm\n"
                         r"$T_e$ = "f"{round(T_e,2)}"r" $\pm$ "f"{round(new_error,2)} ev")
    data = mlines.Line2D([],[],color="blue", marker="o", linestyle="none", label="Masked data")
    plt.legend(handles=[gaussian_line, data,info])        
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.show()

def plot_looper(i,R, T_e_values):

    uncropped = "no"
    super_gaus = "yes"

    if uncropped == "yes":

        plt.scatter(R, T_e_values, s=10)
        plt.xlabel("Radius (m)")
        plt.ylabel("Electron temperature (eV)")
        plt.axhline(500, linestyle="--", color="red")
        temp_crop = mlines.Line2D([],[], linestyle="--", color="red", label="Crop above 500eV")
        plt.text(0.8, 1000, "cutoff", color="red")
        plt.text(0.6, 1000, "---------->", color="red", rotation=90)
        plt.text(1.1, 1000, "---------->", color="red", rotation=90)
        plt.legend(handles=[temp_crop])
        plt.show()

    if super_gaus == "no":

        plt.scatter(R, T_e_values, s=10)
        plt.xlabel("Radius (m)")
        plt.ylabel("Electron temperature (eV)")
        print(f"error: {T_e_error}\n\n")
        plt.ylim(0,500)
        plt.show()

    elif super_gaus == "yes":

        for values in T_e_values: 
            print(values)
            if values > 500:
                position = np.where(T_e_values == values)
                T_e_values = np.delete(T_e_values, position)
                R = np.delete(R, position)



        n_values = [1,2,3,4,5]
        guess = [400,0.9, 0.7]
        for n in n_values:
            def supergaussian(x, *params):
                a = params[0]
                b = params[1]
                c = params[2]
                

                return a * np.exp((-((x-b)**2)/(2*c**2))**n)

            popt, pcov = curve_fit(supergaussian, R, T_e_values,p0=guess, maxfev=50000) 
            supergaussian_fit = supergaussian(R, *popt)
            super_error = np.sqrt(np.diag(pcov))

            if n ==3:

                plt.plot(R, supergaussian_fit, label=f"\nRank: {n}\n"
                         r"$T_{e,MAX}$ = "f"{round(popt[0],2)}" r" $\pm$ "f"{round(super_error[0],2)} eV\n"
                         r"$\sigma_R$ = "f"{round(np.abs(popt[2]),3)}" r" $\pm$ " f"{round(super_error[2],3)} m\n")
            else:
                plt.plot(R, supergaussian_fit, label=str(f"Rank: {n}"))


        plt.scatter(R, T_e_values, s=10)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xlabel("Radius (m)")
        plt.ylabel("Electron temperature (eV)")
        plt.tight_layout()
        plt.show()







if __name__ == "__main__":


    if looper == "yes":
        iter_values = range(0,284)

        T_e_values = np.array([])
        T_e_error = np.array([])
        sigma_lambda_values = np.array([])
        sigma_lambda_error = np.array([])

        for i in iter_values:

            lambda_data, intensity_data = mask(i,L,I)
            gaussian_fit, popt, error = fit(i)
            T_e, new_error = electron_temp_calc(i)

            T_e_values = np.append(T_e_values, T_e)
            T_e_error = np.append(T_e_error, new_error)
            sigma_lambda_values = np.append(sigma_lambda_values, popt[2])
            sigma_lambda_error = np.append(sigma_lambda_error, error)

        plot_looper(i,R, T_e_values)









    






