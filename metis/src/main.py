#!/usr/bin/env python3

import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import os

plot = True

path = "/home/dalir/Documents/cdt/diagnostics_lab/mcf/diagnostics_lab/metis/docs/data/nbi_sweep_failed"

files = os.listdir(f"{path}/mat_files/")
sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))

print(sorted_files)
param_list = ["ne0", "te0", "taue","betap","modeh", "frnbi","pnbi"]
names = [r"Electron number density (m$^{-3}$)", 
         "Electron temperature (ev)",
         "Confinement time (s)",
         r"Plasma beta ($\beta$)",
         "H-mode activation",
         "Fraction of NBI absorbed"]


output_list = np.array([])

"""
def list_subsections():
    print("subsections (I'm using zerod by default):")
    print(full_dataset['post'].dtype)


def list_indexes(subsection='zerod'):
    print("indexes in subsection " + subsection + ":")
    print(full_dataset['post']['zerod'][0][0].dtype)
"""
def get_variable(index,data,subsection="zerod"):
    a = data['post'][subsection][0][0][index][0][0]
    a = [float(x[0]) for x in a]
    return a
   
def get_average(start, end, index,a,subsection='zerod'):
    return (np.mean(a[start:end]), np.std(a[start:end]))

start = 50
end = 100

file_list = np.array([])
for param in param_list:
    for files in sorted_files:
        data = scipy.io.loadmat(f"{path}/mat_files/{files}")
        a = get_variable(param, data)
        average = get_average(start, end,param,a)[0]
        output_list = np.append(output_list, average)

output_list = np.array_split(output_list, len(param_list))


nbi_absorbed = output_list[6]/1000000
lawson = np.array(output_list[0]*output_list[1]*output_list[2])
i = -1
for array in output_list:
    i = i + 1 
    if i > 5:
        break


    plt.plot(nbi_absorbed, array)
    plt.scatter(nbi_absorbed,array)
    plt.xlabel("NBI power absorbed (MW)")
    plt.ylabel(names[i])
    plt.title(f"{names[i]} against NBI power (MW) with other heating")
    plt.tight_layout()
    plt.savefig(f"{path}/pics/{param_list[i]}.png")
    plt.close()
    

plt.scatter(nbi_absorbed, lawson)
plt.plot(nbi_absorbed, lawson)
plt.title(r"Triple product (eVsm$^-{3}$) against NBI power (MW) with other heating")
plt.xlabel("NBI power absorbed (MW)")
plt.ylabel(r"Triple product (eVsm$^-{3})$")
plt.tight_layout()
plt.savefig(f"{path}/pics/lawson.png")
plt.close()


