#!/usr/bin/python

import sys
import re
import math
import numpy
import subprocess
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib import rc
rc('text', usetex=True) # this is if you want to use latex to print text. If you do you can create strings that go on labels or titles like this for example (with an r in front): r"$n=$ " + str(int(n))
from numpy import *
from pylab import *
import random
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.lines as lns

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from common import spaces, get_prod_spaces

prod_spaces = get_prod_spaces([0,3,4])

real_flag = 0 
proc_flag = 0 
flpins_flag = 0
mflops_flag = 0
loop_flag = 0

ALL_CORES = 12
SOCKETS = 2

argl = len(sys.argv)

if argl < 2:
    print "The input file is missing. Please provide the kernel log file with the measurements."
    exit(1)

f_in = sys.argv[1]
type_of_test = "" #sys.argv[2] if argl > 2 else "RHS"
unord = int(sys.argv[3]) if argl > 3 else 0
vectorized = int(sys.argv[4]) if argl > 4 else 1
unord_str = "" if unord < 1 else "_u"
f_out = sys.argv[5] if argl > 5 else f_in.split(".")[0] + unord_str + ".pdf"
f_out1 = sys.argv[5] if argl > 5 else f_in.split(".")[0] + unord_str + "_layers" + ".pdf"

cores = 12
actual_cores = 12
div_factor = 1.0

ln_width = 2
xdist = 1.05

print "Get Extruded meshes PAPI data."
print "Roofline output goes to: ", f_out
print "Half-Roofline output goes to: ", f_out1
lines = open(f_in,'r').read().split("\n") # Open file, read the whole file, split the file in lines 

# Line example:
# 0          1                           2           3           4          5                6        7  8  9  10            11     12     13        
# Kernel bilateralFilterKernel_76800 Real_time 0.3396959901 Proc_time 0.3386929929 Total_instructions 0 IPC 0 Total_flpins 7961999 MFLOPS 784.9369507

# For extruded meshes
# Kernel CG1xCG1_1 Real_time 0.0119426 Proc_time 0.0119426 Total_instructions 0 IPC 0 Total_flpins 12.7448 MFLOPS 1.07161 DV 76.8288 
# Pay attention Total_flpins is Mega

real = {}
proc = {}
ins = {}
ipc = {}
flpins = {}
mflops = {}
names=[]
dv = {} # Data volume (back-of-the-envelope computation)
if loop_flag == 0 :
    for line in lines :
        matching = re.match("^Kernel(.*)",line)
        if matching :
            tokens = line.split(" ")
            name = tokens[1]
            short_name = name.split("_")[0]
            if not(short_name in names):
                names.append(name.split("_")[0])
            if not name in mflops :
                real[name] = []
                proc[name] = []
                flpins[name] = []
                mflops[name] = []
                dv[name] = []
            real[name].append(float(tokens[3]))
            proc[name].append(float(tokens[5]))
            flpins[name].append(float(tokens[11])*pow(10,6))
            mflops[name].append(float(tokens[13]))
            dv[name].append(float(tokens[15])*1024*1024) # Convert it in bytes
            actual_cores = int(tokens[17])
            if float(tokens[17]) < ALL_CORES and cores == ALL_CORES:
                cores = int(ALL_CORES / SOCKETS)
                div_factor = 2.0

totalFlpins = 0
totalFlops = 0
for variable in flpins.keys() :
    totalFlpins += sum(flpins[variable])
    totalFlops += sum(mflops[variable])
totalFlops *= pow(10,6)
   
###########################################
######### Plot the roofline model #########
###########################################

fontsize0=32
fontsize1=28
fontsize2=32
fontsize3=17
fontsize4=18

background_color = 'white'
grid_color = 'black' 

matplotlib.rc('axes', facecolor = background_color)
matplotlib.rc('axes', edgecolor = grid_color)
matplotlib.rc('axes', linewidth = ln_width)
matplotlib.rc('axes', grid = True)
matplotlib.rc('axes', axisbelow = True)
matplotlib.rc('grid', color = grid_color)
matplotlib.rc('grid', linestyle=':')
matplotlib.rc('grid', linewidth=0.5)
matplotlib.rc('grid', alpha=0.3)
matplotlib.rc('legend', fontsize=25)
matplotlib.rc('xtick.major', size =0)
matplotlib.rc('xtick.minor', size =0)
matplotlib.rc('ytick.major', size =0)
matplotlib.rc('ytick.minor', size =0)
colors=["blue", "royalblue", "purple", "magenta", "#A52A2A", "red", "#006400", "#B8860B", "#008B8B", "#FF00FF", "#FF1493"]

# Foraker
# FLP mix included 
SP_singleCore = 2*pow(10,9)
SP_multicore = SP_singleCore * cores
SP_SIMD_multicore = SP_multicore * 4 # double precision 
SP_FP_balance = SP_SIMD_multicore / 0.50	
LLC_B = 15*1024*1024

X_MIN=0.1
X_MAX=30

X_MIN_LAYERS = -5
X_MAX_LAYERS = 110

Y_MIN=pow(10,9)
Y_MAX=4*pow(10,11)

PEAK_PERF_GF = SP_singleCore / pow(10,9)
PEAK_PERF_F = SP_singleCore

PEAK_BW_BS=55872.6 / div_factor*1024*1024 # B/s
PEAK_BW_GBS= 55872.6 / div_factor / 1024 # GB/s

# BW provided by Intel
PEAK_BW_BS_Intel=42600.0*SOCKETS*1024*1024 # B/s
PEAK_BW_GBS_Intel= 42600.0*SOCKETS / 1024 # GB/s
ASPECT_RATIO=1

LOG_X=1
LOG_Y=1

peak_vbw = "%.5s" % PEAK_BW_GBS

fig = plt.figure(figsize=(15, 15))

flops = 0
opIntensity = 0
y = []
x = {}
pointColour = 'blue'
i = 0
min_flops = SP_FP_balance
max_flops = 0
for variable in mflops.keys():
    mem = dv[variable][0] 
    #flops = numpy.mean(mflops[variable], dtype=numpy.float64) * numpy.mean(proc[variable], dtype=numpy.float64) #* pow(10, 6)
    flops = flpins[variable][0]#numpy.mean(mflops[variable]) * numpy.mean(proc[variable]) * pow(10, 6)
    #print flops, "\t",  numpy.mean(mflops[variable]), "\t",  numpy.mean(proc[variable])
    opIntensity = flops / mem
    # layers = 
    tokens = str(variable).split("_")
    name = tokens[0]
    size = tokens[1]
    tokens = name.split("Kernel")
    nameCleaned = tokens[0]
    #x.append(opIntensity)
    if mem - LLC_B > 0:
        for i, space in enumerate(prod_spaces):
            if space == name:
                pointColour = colors[i]
                break
    else: 
        pointColour = 'green'
    if not nameCleaned in x:
        x[nameCleaned] = []
    value = numpy.mean(mflops[variable]) * pow(10, 6)
    if min_flops > value:
        min_flops = value
    if max_flops < value:
        max_flops = value
    x[nameCleaned].append([opIntensity, "", size, pointColour, 0, value, numpy.sum(flpins[variable])])
    i += 1

Y_MIN = min_flops * 0.95
if not vectorized:
    Y_MIN *= 0.85

if actual_cores == 1:
    if vectorized:
        Y_MAX = 8.0 * SP_singleCore * xdist * 1.15
    else:
        Y_MAX = 2.0 * SP_singleCore * xdist * 1.15
    Y_MAX = max(Y_MAX, 2 * 1e10)
else:
    if vectorized:
        Y_MAX = SP_FP_balance * xdist * 1.19
    else:
        Y_MAX = 2.0 * SP_multicore * xdist * 1.19
    Y_MAX = max(Y_MAX, max(Y_MIN, 1e10) * 10)

# SETUP ROOFLINE MODEL
# Returns the Axes instance
ax = fig.add_subplot(111, aspect=ASPECT_RATIO)

#x-y range
ax.axis([X_MIN,X_MAX,Y_MIN,Y_MAX])

ffont = {'family':'sans-serif','fontsize':fontsize2,'weight':'bold'}
ax.set_xticklabels(ax.get_xticks(),ffont)
ax.set_yticklabels(ax.get_yticks(),ffont)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)

#Log scale
if LOG_Y: ax.set_yscale('log')
if LOG_X: ax.set_xscale('log', basex=2)

#formatting:
ax.set_title("E5-2620 Xeon Sandy Bridge EP %(type)s" % {'type': ""}, fontsize=fontsize0, fontweight='bold')
ax.set_xlabel("Operational Intensity [FLOPs/Byte]", fontsize=fontsize1)
ax.set_ylabel("Performance [FLOPS]", fontsize=fontsize1)

i = 0
ii = 0
COM_Int = 0
COM_Flops = 0 
for variable in x:
    aux_x = []
    aux_y = []
    x[variable] = sorted(x[variable], key=lambda x: int(x[2]))
    for var2 in range(0, len(x[variable])):
        tokens = str(variable).split("_")
        i += 1
        aux_x.append(x[variable][var2][0])
        aux_y.append(x[variable][var2][5])
        if int(x[variable][var2][2]) == 0:
            ax.scatter([aux_x[var2],],[aux_y[var2],], s=100, color=x[variable][0][3], alpha=1.0)
        else:
            ax.scatter([aux_x[var2],],[aux_y[var2],], s=80, color=x[variable][0][3], alpha=1.0)
        xOffset = 0  
        yOffset = 0
        # ax.annotate(x[variable][var2][2] if int(x[variable][var2][2]) > 0 else "", xy=(aux_x[var2], aux_y[var2]), xycoords='data', xytext=(-15+xOffset, -10+yOffset), textcoords='offset points', fontsize=fontsize4) 
    if len(x[variable]) != 1 :
        for ii, space in enumerate(prod_spaces):
            if space == tokens[0] and space in names:
                pickedColour = colors[ii]
    if int(x[variable][0][2]) == 0:
        plt.plot(aux_x[1:], aux_y[1:], '-', color=pickedColour, linewidth=ln_width+1)
    else:
        plt.plot(aux_x, aux_y, '-', color=pickedColour, linewidth=ln_width+1)
    xOffset = 0
    yOffset = 0
    #### 1 annotation for a group of points
    ax.annotate(x[variable][0][1], xy=(aux_x[0], aux_y[0]), xycoords='data', xytext=(+3+xOffset, +1+yOffset), textcoords='offset points', fontsize=fontsize4)

# This points are only for the legend (change this in the future)
cnt = 0
for i, space in enumerate(prod_spaces):
    if space in names:
        cnt += 1
        ax.scatter([-cnt,],[-cnt,], s=80, color=colors[i], alpha=1.0, label=space)
gca().legend(loc="lower right")

# alpha should differ
low_alpha = 0
alpha_gflops = low_alpha

pos = 1

if actual_cores == 1 and not vectorized:
    alpha_gflops = 1.0
#Peak performance lines and texts
if Y_MIN < SP_singleCore:
    ax.axhline(y=SP_singleCore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax.text(pos, SP_singleCore * xdist, "Monocore ("+str(float(SP_singleCore)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=SP_singleCore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    # ax1.text(4, SP_singleCore * xdist, "Monocore ("+str(float(SP_singleCore)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)


if Y_MIN < 2.0 * SP_singleCore:
    ax.axhline(y=2*SP_singleCore, linewidth=ln_width, color='red', alpha=alpha_gflops)
    #ax.text(pos, 2.0*SP_singleCore * xdist, "Monocore + FPB ("+str(float(SP_singleCore) * 2.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=2*SP_singleCore, linewidth=ln_width, color='red', alpha=alpha_gflops)
    # ax1.text(4, 2.0*SP_singleCore * xdist, "Monocore + FPB ("+str(float(SP_singleCore) * 2.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)


if actual_cores == 1 and vectorized:
    alpha_gflops = 1.0
    ax.axhline(y=4.0*SP_singleCore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax.text(pos, 4.0*SP_singleCore * xdist, "SIMD ("+str(float(SP_singleCore) * 4.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    ax.axhline(y=8.0*SP_singleCore, linewidth=ln_width, color='red', alpha=alpha_gflops)
    #ax.text(pos, 8.0*SP_singleCore * xdist, "BALANCE ("+str(float(SP_singleCore) * 8.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=4.0*SP_singleCore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    # ax1.text(4, 4.0*SP_singleCore * xdist, "SIMD ("+str(float(SP_singleCore) * 4.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=8.0*SP_singleCore, linewidth=ln_width, color='red', alpha=alpha_gflops)
    # ax1.text(4, 8.0*SP_singleCore * xdist, "SIMD + FPB ("+str(float(SP_singleCore) * 8.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)


if actual_cores > 1 and not vectorized:
    alpha_gflops = 1.0
else:
    alpha_gflops = low_alpha

if Y_MIN < SP_multicore and actual_cores > 1:
    ax.axhline(y=SP_multicore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax.text(pos + 2, SP_multicore * xdist, "TLP ("+str(float(SP_multicore)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=SP_multicore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    # ax1.text(4, SP_multicore * xdist, "TLP ("+str(float(SP_multicore)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

if Y_MIN < 2.0 * SP_multicore and actual_cores > 1:
    ax.axhline(y=2*SP_multicore, linewidth=ln_width, color='red', alpha=alpha_gflops)
    #ax.text(pos, 2.0 * SP_multicore * xdist, "BALANCE ("+str(float(SP_multicore) * 2.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=2*SP_multicore, linewidth=ln_width, color='red', alpha=alpha_gflops)
    # ax1.text(4, 2.0 * SP_multicore * xdist, "TLP + FPB ("+str(float(SP_multicore) * 2.0/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)


if actual_cores > 1 and vectorized:
    alpha_gflops = 1.0
    ax.axhline(y=SP_SIMD_multicore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax.text(pos, SP_SIMD_multicore * xdist, "SIMD ("+str(float(SP_SIMD_multicore)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    ax.axhline(y=SP_FP_balance, linewidth=ln_width, color='red', alpha=alpha_gflops)
    #ax.text(pos, SP_FP_balance * xdist, "BALANCE ("+str(float(SP_FP_balance)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=SP_SIMD_multicore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    # ax1.text(4, SP_SIMD_multicore * xdist, "SIMD ("+str(float(SP_SIMD_multicore)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    # ax1.axhline(y=SP_FP_balance, linewidth=ln_width, color='red', alpha=alpha_gflops)
    # ax1.text(4, SP_FP_balance * xdist, "SIMD + FPB ("+str(float(SP_FP_balance)/pow(10,9))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

#BW line and text
xx = np.linspace(X_MIN, X_MAX, 100)
y = xx * PEAK_BW_BS
ax.plot(xx, y, linewidth=ln_width, color='black')

if actual_cores == 1:
    ax.text(0.125, PEAK_BW_BS * 0.56, 'STREAM BW ('+str(peak_vbw)+' GB/sec)', fontsize=fontsize2, rotation=45)
else:
    ax.text(0.125, PEAK_BW_BS * 1.0, 'STREAM BW ('+str(peak_vbw)+' GB/sec)', fontsize=fontsize2, rotation=45)
#ax.text(Y_MAX/PEAK_BW_BS,Y_MAX*7/10,'MemBandwidth ('+str(PEAK_BW_GBS)+' GB/sec)',fontsize=fontsize2)

#save file
fig.savefig(f_out, dpi=250,  bbox_inches='tight')

# ===========================================================================
# SETUP HALF ROOFLINE MODEL
fig1 = plt.figure(figsize=(15, 15))

SP_SIMD_multicore /= 1e9
SP_multicore /= 1e9
SP_FP_balance /= 1e9
SP_singleCore /= 1e9

Y_MIN = 0

if actual_cores == 1:
    if vectorized:
        Y_MAX = 8.0 * SP_singleCore * xdist * 1.1
    else:
        Y_MAX = 2.0 * SP_singleCore * xdist * 1.1
elif not vectorized:
    Y_MAX = 2.0 * SP_multicore * xdist * 1.1
else:
    Y_MAX = SP_FP_balance * xdist * 1.1

ax1 = fig1.add_subplot(111)
ax1.axis([X_MIN_LAYERS, X_MAX_LAYERS, Y_MIN, Y_MAX])

ffont = {'family':'sans-serif','fontsize':fontsize2,'weight':'bold'}
ax1.set_xticklabels(ax1.get_xticks(), ffont)
ax1.set_yticklabels(ax1.get_yticks(), ffont)

for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)

#Log scale
# if LOG_Y: ax1.set_yscale('log')
# if LOG_X: ax1.set_xscale('log', basex=2)
#formatting:
ax1.set_title("E5-2620 Xeon Sandy Bridge EP %(type)s" % {'type': ""}, fontsize=fontsize0, fontweight='bold')
ax1.set_xlabel("Number of layers", fontsize=fontsize1)
ax1.set_ylabel("Performance [GFLOPS]", fontsize=fontsize1)

i = 0
ii = 0
COM_Int = 0
COM_Flops = 0 
for variable in x:
    aux_layers_x = []
    aux_layers_y = []
    x[variable] = sorted(x[variable], key=lambda x: int(x[2]))
    for var2 in range(0, len(x[variable])):
        tokens = str(variable).split("_")
        i += 1
        if int(x[variable][var2][2]) == 0:
            aux_layers_x.append(1)
        else:
            aux_layers_x.append(int(x[variable][var2][2]))
        aux_layers_y.append(x[variable][var2][5] / 1e9)
        xOffset = 0  
        yOffset = 0
        if int(x[variable][var2][2]) == 0:
            ax1.scatter([aux_layers_x[var2],],[aux_layers_y[var2],], s=140, color=x[variable][0][3], alpha= 1.0, marker="^")
        else:
            ax1.scatter([aux_layers_x[var2],],[aux_layers_y[var2],], s=80, color=x[variable][0][3], alpha= 1.0)

    if len(x[variable]) != 1:
        for ii, space in enumerate(prod_spaces):
            if space == tokens[0] and space in names:
                pickedColour = colors[ii]
    if int(x[variable][0][2]) == 0:
        plt.plot(aux_layers_x[1:], aux_layers_y[1:], '-', color=pickedColour, linewidth=ln_width+1)
    else:
        plt.plot(aux_layers_x, aux_layers_y, '-', color=pickedColour, linewidth=ln_width+1)  

# This points are only for the legend (change this in the future)
cnt = 0
for i, space in enumerate(prod_spaces):
    if space in names:
        cnt += 1
        ax1.scatter([-cnt,],[-cnt,], s=80, color=colors[i], alpha=1.0, label=space)
gca().legend(loc="lower right")

# alpha should differ
low_alpha = 0
alpha_gflops = low_alpha

pos = 50

if actual_cores == 1 and not vectorized:
    alpha_gflops = 1.0
#Peak performance lines and texts
if Y_MIN < SP_singleCore:
    ax1.axhline(y=SP_singleCore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax1.text(pos, SP_singleCore * xdist, "1 CORE ("+str(float(SP_singleCore))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)


if Y_MIN < 2.0 * SP_singleCore:
    ax1.axhline(y=2*SP_singleCore, linewidth=ln_width, color='red', alpha=low_alpha)
    #ax1.text(pos, 2.0*SP_singleCore * xdist, "BALANCE ("+str(float(SP_singleCore) * 2.0)+" GFLOPS)", fontsize=fontsize2, alpha=low_alpha)


if actual_cores == 1 and vectorized:
    alpha_gflops = 1.0
    ax1.axhline(y=4.0*SP_singleCore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax1.text(pos, 4.0*SP_singleCore * xdist, "SIMD ("+str(float(SP_singleCore) * 4.0)+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    ax1.axhline(y=8.0*SP_singleCore, linewidth=ln_width, color='red', alpha=low_alpha)
    #ax1.text(pos, 8.0*SP_singleCore * xdist, "BALANCE ("+str(float(SP_singleCore) * 8.0)+" GFLOPS)", fontsize=fontsize2, alpha=low_alpha)


if actual_cores > 1 and not vectorized:
    alpha_gflops = 1.0
else:
    alpha_gflops = low_alpha

if Y_MIN < SP_multicore and actual_cores > 1:
    ax1.axhline(y=SP_multicore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax1.text(pos, SP_multicore * xdist, str(cores) + " CORES ("+str(float(SP_multicore))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

if Y_MIN < 2.0 * SP_multicore and actual_cores > 1:
    ax1.axhline(y=2*SP_multicore, linewidth=ln_width, color='red', alpha=low_alpha)
    #ax1.text(pos, 2.0 * SP_multicore * xdist, "BALANCE ("+str(float(SP_multicore) * 2.0)+" GFLOPS)", fontsize=fontsize2, alpha=low_alpha)


if actual_cores > 1 and vectorized:
    alpha_gflops = 1.0
    ax1.axhline(y=SP_SIMD_multicore, linewidth=ln_width, color='black', alpha=alpha_gflops)
    #ax1.text(pos, SP_SIMD_multicore * xdist, "SIMD ("+str(float(SP_SIMD_multicore))+" GFLOPS)", fontsize=fontsize2, alpha=alpha_gflops)

    ax1.axhline(y=SP_FP_balance, linewidth=ln_width, color='red', alpha=low_alpha)
    #ax1.text(pos, SP_FP_balance * xdist, "BALANCE ("+str(float(SP_FP_balance))+" GFLOPS)", fontsize=fontsize2, alpha=low_alpha)

fig1.savefig(f_out1, dpi=250,  bbox_inches='tight')

