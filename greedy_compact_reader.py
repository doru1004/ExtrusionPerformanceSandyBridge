import sys
import math
from firedrake import *
import pyop2 as op2
from pyop2.record import *

# Calling args
# python greedy_compact_reader.py f_in f_out f_extra

# Read in parameters
argl = len(sys.argv)
f_in = sys.argv[1]
out_dir = sys.argv[2] if 2 < argl else ""
form_side = sys.argv[3] if 3 < argl else ""
vectorized = int(sys.argv[4]) if 4 < argl else 1
likwid_dirs = [sys.argv[5]] if 5 < argl else []
# Frequency
frequency = 2.0

STREAM_BW = 55300.0
TLP_FPB = 24
# Hardcoded at the moment, these are only vaild for v*dx
adds_to_muls = {}
adds_to_muls["CG1xCG1"] = 1.43
adds_to_muls["CG1xDG0"] = 1.46
adds_to_muls["CG1xDG1"] = 1.43
adds_to_muls["DG0xCG1"] = 1.47
adds_to_muls["DG0xDG0"] = 1.48
adds_to_muls["DG0xDG1"] = 1.47
adds_to_muls["DG1xCG1"] = 1.43
adds_to_muls["DG1xDG0"] = 1.46
adds_to_muls["DG1xDG1"] = 1.43

fv = {}
fv["CG1xCG1"] = 1.00
fv["CG1xDG0"] = 1.00
fv["CG1xDG1"] = 1.00
fv["DG0xCG1"] = 1.00
fv["DG0xDG0"] = 1.00
fv["DG0xDG1"] = 1.00
fv["DG1xCG1"] = 1.00
fv["DG1xDG0"] = 1.00
fv["DG1xDG1"] = 1.00

if form_side == "FRHS":
    # This is for fvdx.
    adds_to_muls = {}
    adds_to_muls["CG1xCG1"] = 1.70
    adds_to_muls["CG1xDG0"] = 1.81
    adds_to_muls["CG1xDG1"] = 1.70
    adds_to_muls["DG0xCG1"] = 1.65
    adds_to_muls["DG0xDG0"] = 1.50
    adds_to_muls["DG0xDG1"] = 1.65
    adds_to_muls["DG1xCG1"] = 1.70
    adds_to_muls["DG1xDG0"] = 1.81
    adds_to_muls["DG1xDG1"] = 1.70

    fv = {}
    fv["CG1xCG1"] = 1.58
    fv["CG1xDG0"] = 1.00
    fv["CG1xDG1"] = 1.58
    fv["DG0xCG1"] = 1.00
    fv["DG0xDG0"] = 1.00
    fv["DG0xDG1"] = 1.00
    fv["DG1xCG1"] = 1.58
    fv["DG1xDG0"] = 1.00
    fv["DG1xDG1"] = 1.58

mpi_procs = [1, 6, 12, 24]
all_layers = [1, 2, 4, 10, 30, 50, 100]

IDEAL_BW = lambda ds, factor: ds[7] / ds[3] * factor * 100 / (ds[7] / ds[16])
VBW = lambda ds, factor: ds[7] / ds[3] * 100 / STREAM_BW
GFLOPS = lambda ds, factor: ds[14] / 1000.0

A2M_RATIO = lambda ds, factor: adds_to_muls[factor]
VECT_RATIO = lambda ds, factor: fv[factor]

A2M_GFLOPS = lambda ds, factor: TLP_FPB * A2M_RATIO(ds, factor) * (4.0 if vectorized else 1.0)
VECT_A2M_GFLOPS = lambda ds, factor: TLP_FPB * A2M_RATIO(ds, factor) * VECT_RATIO(ds, factor)

A2M_PERC_PEAK = lambda ds, factor: GFLOPS(ds, factor) * 100.0 / A2M_GFLOPS(ds, factor)
VECT_A2M_PERC_PEAK = lambda ds, factor: GFLOPS(ds, factor) * 100.0 / VECT_A2M_GFLOPS(ds, factor)

VBW_PEAK = lambda ds, factor: ds[10] * 100.0 / STREAM_BW

class GeneratePlots(object):
    """ This is the class that will generate the data files and plot scripts for gnuplot.
    """
    space = [('CG', 1), ('CG', 2), ('CG', 3), ('DG', 0), ('DG', 1), ('DG', 2), ('DG', 3)]
    # Order of the measures in a self._data record.
    # 0  threads
    # 1  layers
    # 2  space_name
    # 3  self.runtime
    # 4  self.rv_runtime
    # 5  self.v_volume
    # 6  self.m_volume
    # 7  self.mv_volume
    # 8  self.vbw
    # 9  self.mbw
    # 10 self.mvbw
    # 11 self.rvbw
    # 12 self.iaca_flops
    # 13 self.papi_flops
    # 14 self.iaca_mflops
    # 15 self.papi_mflops
    # 16 self.cycles
    vset2str = ["MPI Processes", "Layers",                # 0, 1
                "Outer Product Discretization",           # 2
                "Runtime of a RHS assembly [s]",          # 3
                "Re-ordered RHS assembly [s]",            # 4
                "Read Valuable Data Volume [MB]",         # 5
                "Maximal (Pessimistic) Data Volume [MB]", # 6
                "Valuable Data Volume [MB]",              # 7
                "Read Valuable BW [MB/s]",                # 8
                "Maximal Valuable BW [MB/s]",             # 9
                "Valuable BW [MB/s]",                     # 10
                "Unordered Valuable BW [MB/s]",           # 11
                "Floating point operations [MFLOP]",      # 12
                "Floating point operations [MFLOP]",      # 13
                "FLOP rate [MFLOPS]",                     # 14
                "FLOP rate [MFLOPS]",                     # 15
                "Idealized runtime [s]",                  # 16
                "Total process time [s]"]
    short_str = ["PROCS", "LAYERS", "DISCR", "RT", "RRT", "VDV", "MDV",
                 "MVDV", "VBW", "MBW", "MVBW", "RVBW", "IACAF", "PAPIF",
                 "IACAMF", "PAPIMF", "CYCLES", "CRT"]

    def __init__(self, f_in, extra_dir_list=[]):
        # These are the sets contatining all the possible values that are contained in the data_space
        # for each field.
        self._vset = []
        # threads
        self._vset.append([1, 6, 12, 24])
        # layers
        self._vset.append([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 25, 30, 32, 48, 50, 64, 80, 96, 100, 128, 144, 160, 176, 192])
        # product spaces
        prod_spaces = []
        for s1 in GeneratePlots.space:
            for s2 in GeneratePlots.space:
                prod_spaces.append(self._fuse(s1, s2))
        self._vset.append(prod_spaces)

        # Mapping between each vset and the column in the data space
        self._vset2ds = [0, 1, 2]
        self._spaces = []
        self._yset_str = ["", "", "", "",
                          "", "", "", "",
                          "-READ-BW", "-MAX-BW", "-VAL-BW", "-UNORD-BW",
                          "", "", "", "",
                          "-IDEAL", "-MEASURED", "-LIKWID-BW", ""]

        # List of folders with extra data
        self._data = []
        reduction_record = ReductionRecord()
        layers = -1
        threads = 0
        horiz = -1
        vert = -1
        with open(f_in, "r") as f1:
            for line in f1:
                words = line.split()
                if "base" in line:
                  reduction_record.add_values(words)
                else:
                    if not reduction_record.isEmpty():
                        # Check to see if there are any extra measures which need to be part of the data
                        extra_vals = self.read_likwid_data(extra_dir_list, horiz, vert, threads, layers)
                        # Add all the data to together
                        self._data += [[threads, layers, space_name] + reduction_record.plot_list(frequency) + extra_vals]
                        # self.read_extra_data(dir_name, field_name)
                        reduction_record = ReductionRecord()
                    if "layers" in line:
                        layers = int(words[2])
                    elif "number of threads" in line:
                        threads = int(words[4])
                        horiz = int(words[5])
                        vert = int(words[6])
                        space_name = GeneratePlots.space[horiz][0] + str(GeneratePlots.space[horiz][1]) + "x" + \
                                     GeneratePlots.space[vert][0] + str(GeneratePlots.space[vert][1])
                        if not space_name in self._spaces:
                            self._spaces += [space_name]

    def read_likwid_data(self, extra_dir_list, horiz, vert, threads, layers):
        extra_vals = []
        for d in extra_dir_list:
            extra_res_dir = d + "/" + str(horiz) + "x" + str(vert) + "x" + str(threads) + "x" + str(layers)
            import os
            if not os.path.exists(extra_res_dir):
                extra_vals += [0.0]
                continue
            value_found = 0.0
            field_found = ""
            for filename in os.listdir(extra_res_dir):
                with open(extra_res_dir+"/"+filename) as h:
                    for line in h:
                        if "Memory BW [MBytes/s]" in line:
                            field_found = "Memory BW [MBytes/s]"
                            value_found = max(value_found, float(line.split()[5]))
                            GeneratePlots.vset2str.append(["Measured MM BW [MB/s]"])
                            GeneratePlots.short_str.append(["MMMBW"])
                        elif "L2 miss ratio" in line:
                            value_found = max(value_found, float(line.split()[5]))
                        elif "L3 miss ratio" in line:
                            value_found = max(value_found, float(line.split()[5]))
                        elif "L2 bandwidth [MBytes/s]" in line:
                            value_found = max(value_found, float(line.split()[5]))
                        elif "L3 bandwidth [MBytes/s]" in line:
                            value_found = max(value_found, float(line.split()[5]))
                        elif "L1 DTLB load misses" in line:
                            value_found = max(value_found, float(line.split()[6]))
                        elif "L1 DTLB store misses" in line:
                            value_found = max(value_found, float(line.split()[6]))
                        elif "L1 DTLB load miss rate" in line:
                            value_found = max(value_found, float(line.split()[7]))
                        elif "L1 DTLB store miss rate" in line:
                            value_found = max(value_found, float(line.split()[7]))
                        elif "L1 ITLB misses" in line:
                            value_found = max(value_found, float(line.split()[5]))
                        elif "L1 ITLB miss rate" in line:
                            value_found = max(value_found, float(line.split()[6]))
            if field_found == "Memory BW [MBytes/s]":
                value_found = value_found * (2.0 if threads in [12, 24] else 1.0)
            extra_vals += [value_found]
        return extra_vals

    # Loop over the graphs and fix the parameter that varies across the graphs
    # Call plot graphs with:
    #  arg1: the data space as the first argument
    #  arg2: Variation across graphs, what each graph will be for
    #  arg3: Variation across graph lines, it's what each line in the graph will mean
    #  arg4: Variation across the X-axis, what the x-axis will contain
    #  arg5: Variation across the Y-axis, what's on the y-axis
    # In this case the last arg is the column in the data_space where the values
    # of the y-axis reside
    def _construct_graph_data(self, g_idx, l_idx, x_idx, y_idx):
        constructed_graphs = []
        # For each parameter we want to vary across graphs
        for g in self._vset[g_idx]:
            # Loops over all the data
            temp_g = Graph(GeneratePlots.vset2str[g_idx],
                           GeneratePlots.vset2str[x_idx],
                           GeneratePlots.vset2str[y_idx],
                           str(g),
                           x_idx)
            for ds in self._data:
                if ds[self._vset2ds[g_idx]] == g:
                    # Now loop over the possible values for the lines
                    for l in self._vset[l_idx]:
                        if ds[self._vset2ds[l_idx]] == l:
                            # Loop over the possible values for the x-axis
                            for x_val in self._vset[x_idx]:
                                if ds[self._vset2ds[x_idx]] == x_val:
                                    # print ds, len(ds), y_idx, GeneratePlots.record_size
                                    # from IPython import embed; embed()
                                    temp_g.add_data(l, x_val, ds[y_idx])
            if not temp_g.empty:
                constructed_graphs.append(temp_g)
        return constructed_graphs

    def _construct_graph_data_hybrid(self, g_idx, l_idx, x_idx, y_str, l_set, y_set):
        constructed_graphs = []
        # For each parameter we want to vary across graphs
        for g in self._vset[g_idx]:
            # Loops over all the data
            temp_g = Graph(GeneratePlots.vset2str[g_idx],
                           GeneratePlots.vset2str[x_idx],
                           y_str,
                           str(g),
                           x_idx)
            for ds in self._data:
                if ds[self._vset2ds[g_idx]] == g:
                    # Now loop over the possible values for the lines
                    for l in self._vset[l_idx]:
                        if ds[self._vset2ds[l_idx]] == l and l in l_set:
                            # Loop over the possible values for the x-axis
                            for x_val in self._vset[x_idx]:
                                if ds[self._vset2ds[x_idx]] == x_val:
                                    for y_set_val in y_set:
                                        temp_g.add_data(l + self._yset_str[y_set_val], x_val, ds[y_set_val])
            if not temp_g.empty:
                constructed_graphs.append(temp_g)
        return constructed_graphs

    def _construct_table(self, g_idx, l_idx, x_idx, g_set, l_set, x_set, FUNCS):
        constructed_graphs = []
        # For each parameter we want to vary across graphs
        for g in self._vset[g_idx]:
            # Loops over all the data
            for ds in self._data:
                if ds[self._vset2ds[g_idx]] == g and (g in g_set or g_set == []):
                    # Now loop over the possible values for the lines
                    for l in self._vset[l_idx]:
                        if ds[self._vset2ds[l_idx]] == l and (l in l_set or l_set == []):
                            # Loop over the possible values for the x-axis
                            for x_val in self._vset[x_idx]:
                                if ds[self._vset2ds[x_idx]] == x_val and x_val == x_set[0]:
                                    factor = 2.0 if x_val == 24 else 1.0
                                    res = []
                                    if isinstance(FUNCS, list):
                                        for f in FUNCS:
                                            res.append("%.5s" % f(ds, l))
                                        return " & ".join(res)
                                    return "%.5s" % FUNCS(ds, factor)

    def _construct_speed_up_table(self, g_idx, l_idx, x_idx, g_set, l_set, x_set, time=17):
        constructed_graphs = []
        # For each parameter we want to vary across graphs
        ref_val = None
        res = []
        for g in self._vset[g_idx]:
            # Loops over all the data
            for ds in self._data:
                if ds[self._vset2ds[g_idx]] == g and (g in g_set or g_set == []):
                    # Now loop over the possible values for the lines
                    for l in self._vset[l_idx]:
                        if ds[self._vset2ds[l_idx]] == l and (l in l_set or l_set == []):
                            # Loop over the possible values for the x-axis
                            for x_val in self._vset[x_idx]:
                                if ds[self._vset2ds[x_idx]] == x_val and x_val == x_set[0]:
                                    if not ref_val and g == 1:
                                        # This is the 1 layer case
                                        ref_val = ds[time]
                                    if not ref_val:
                                        raise RuntimeError("Ref_val is unset.")
                                    res += ["%.4s" % (ref_val * 1.0 / ds[time])]
        return " & ".join(res) + " \\\\\\hline"

    def _output_to_roofline(self, filename, unord=False):
        line = "Kernel %(kernel)s Real_time %(rtime)s Proc_time %(ptime)s Total_instructions %(instr)s IPC %(ipc)s Total_flpins %(flpins)s MFLOPS %(mflops)s DV %(dv)s procs %(procs)s\n"
        # Loop over layers
        unord_str = "_u" if unord else ""
        for discr in ["CG1xCG1", "CG1xDG0", "CG1xDG1", "DG0xCG1", "DG0xDG0", "DG0xDG1", "DG1xCG1", "DG1xDG0", "DG1xDG1"]:
            for threads in [1, 6, 12, 24]:
                with open(filename + "_" + discr + "_" + str(threads) + unord_str + ".txt", "w+") as h:
                    for layers in [1, 2, 4, 10, 30, 50, 100]:
                        for d in self._data:
                            if d[2] == discr and d[1] == layers and d[0] == threads:
                                h.write(line % {'kernel': discr + "_" + str(layers) + "_" + str(threads),
                                                'rtime': str(d[3]) if not unord else str(d[4]),
                                                'ptime': str(d[3]) if not unord else str(d[4]),
                                                'instr': str(0),
                                                'ipc': str(0),
                                                'flpins': str(d[12]),
                                                'mflops': str(d[14]) if not unord else str(d[12] / d[4]),
                                                'dv': str(d[7]),
                                                'procs': str(threads)})
                                if layers == 1 and not unord:
                                    h.write(line % {'kernel': discr + "_" + str(0) + "_" + str(threads),
                                                    'rtime': str(d[4]),
                                                    'ptime': str(d[4]),
                                                    'instr': str(0),
                                                    'ipc': str(0),
                                                    'flpins': str(d[12]),
                                                    'mflops': str(d[12] / d[4]),
                                                    'dv': str(d[7]),
                                                    'procs': str(threads)})
                                break

    def _fuse(self, s1, s2):
        return s1[0] + str(s1[1]) + "x" + s2[0] + str(s2[1])

class Graph(object):
    """ Graph object for holding the data and metadata of a graph."""

    linewidth = 3
    lt = [2, 3, 4, 5, 1, 6, 7, 8, 9, 10]
    col = ["blue", "royalblue", "purple", "magenta", "#A52A2A", "red", "#FF00FF", "#FF1493", "#006400", "#B8860B", "#008B8B"]

    def __init__(self, graph_str, x_str, y_str, graph_val, x_id):
        self._x_str = x_str
        self._y_str = y_str
        self._graph_str = graph_str
        self._graph_val = graph_val
        self._x_id = x_id
        self._values = dict()
        self._keys = []
        self._max_xval = 0.0

    def add_data(self, line, x_val, y_val):
        if not(line in self._values):
            self._values[line] = []
        found_key = False
        for k in self._keys:
            if k == str(line):
                found_key = True
        if not found_key:
            self._keys.append(str(line))
        if self._max_xval < x_val:
            self._max_xval = x_val
        self._values[str(line)].append((x_val, y_val))

    @property
    def title(self):
        """Title of the graph."""
        return self._graph_val + " " + self._graph_str

    @property
    def x_label(self):
        """Label of the x-axis"""
        return self._x_str

    @property
    def y_label(self):
        """Label of the x-axis"""
        return self._y_str

    @property
    def empty(self):
        """Returns True if the graph is empty."""
        return len(self._values) == 0

    def plot(self, f_out, out_dir):
        """
        Output the contents to gnuplot file.
        
        arg1: The file is just a filename without extension. 
        """
        gnuplot_graph = ""
        x_axis_vals = []

        f_out += "_" + str(self._graph_val)

        for i, key in enumerate(self._keys):
            vals = self._values[key]
            if i > 0:
                gnuplot_graph += ", "
            for v in vals:
                if not(v[0] in set(x_axis_vals)):
                    x_axis_vals.append(v[0])
            gnuplot_graph += " \"" + out_dir + f_out + ".dat" + "\" using 1:%(column)s with linespoints linewidth %(lw)s lt %(lt)s lc rgb \"%(col)s\" title \""% \
                             {'lt': str(Graph.lt[i % len(Graph.lt)]),
                              'column': str(i+2),
                              'col': Graph.col[i % len(Graph.col)],
                              'lw': Graph.linewidth} + key + "\""

        with open(out_dir+f_out+".dat", "w+") as h:
            for i, x_val in enumerate(x_axis_vals):
                line = str(x_val) + " "
                for j, key in enumerate(self._keys):
                    vals = self._values[key]
                    found = False
                    for v in vals:
                        if v[0] == x_val:
                            line += str(v[1]) + " "
                            found = True
                    if not found:
                        line += "0.0 "
                h.write(line+"\n")
        self._generate_gnuplot_script(f_out, out_dir, gnuplot_graph)
        return gnuplot_graph

    def _generate_gnuplot_script(self, f_out, out_dir, gnuplot_graph):
        _xrange, _position = self._compute_xrange()
        with open(out_dir + f_out + ".gnuplot", "w+") as h:
            h.write("""
#!/usr/bin/env gnuplot

set terminal pdfcairo enhanced color font 'Arial'

set autoscale
set termoption dash
set key outside %(position)s

%(xrange)s

set xlabel \"{/=16 %(x_label)s}\"
set ylabel \"{/=16 %(y_label)s}\"

              """ % {'x_label': self._escape_underscore(self.x_label),
                     'y_label': self._escape_underscore(self.y_label),
                     'xrange': _xrange,
                     'position': _position})
            h.write("\nset output \"%(out)s_plot.pdf\"\n" % {'out': out_dir + f_out})
            h.write("\nset title \"{/=14 %(title)s}\"\n" % {'title': self._escape_underscore(self.title)})
            h.write("plot " + gnuplot_graph)

    def _compute_xrange(self):
        if self._x_id == 0:
            # If the x axis needs to plot the threads
            return "set xrange[0:26]", "right"
        if self._x_id == 1:
            # If the x axis plots the layers
            return "set xrange[0:%(layer_bound)s]" % {'layer_bound': self._max_xval}, "right"
        return "", "right"

    def _escape_underscore(self, s):
        return " ".join([word for word in s.split("_")])

def print_table(FUNC, mpi_ps=mpi_procs, layer_list=all_layers):
    # Print tables containing the percentage of ideal VBW achieved.
    for p in mpi_ps:
        print "Processes: ", p
        for line_val in gen_plots._spaces:
            val = []
            for layers in layer_list:
                val += [gen_plots._construct_table(graph_idx, 2, x_axis_idx, [layers], [line_val], [p], FUNC)]
            if line_val and (len(val) > 0 and val[0]):
                h_times_v = "$" + " \\mathrm{" + line_val.replace("x", "} \\times \\mathrm{") + "}" +  "$"
                print " & ".join([h_times_v] + val) + "\\\\\\hline"

gen_plots = GeneratePlots(f_in, likwid_dirs)

# Plot a variety of graphs
# graph_idx: what goes on top of the graph
# line_idx: what goes on each line
# 0  threads
# 1  layers
# 2  space_name
# 3  self.runtime
# 4  self.rv_runtime
# 5  self.v_volume
# 6  self.m_volume
# 7  self.mv_volume
# 8  self.vbw
# 9  self.mbw
# 10 self.mvbw
# 11 self.rvbw
# 12 self.iaca_flops
# 13 self.papi_flops
# 14 self.iaca_mflops
# 15 self.papi_mflops
# 16 self.cycles
# 17 self.c_runtime
measures = [3, 4, 8, 9, 10, 11, 12, 14]
mem_set = [8, 9, 10, 11]
if len(likwid_dirs) > 0:
    for i in range(len(likwid_dirs)):
        measures += [18 + i]
    mem_set += [18]

# Also plot hybrid graphs when some of the measures have the same type.
# For example in the case of BW there are several types of BW which can be plotted on the same graph.
graph_idx = 1
x_axis_idx = 0
for line_val in gen_plots._spaces:
    f_out = gen_plots.short_str[2] + "-" + line_val + "-" + \
            "BW" + "_vs_" + \
            gen_plots.short_str[x_axis_idx]
    gs = gen_plots._construct_graph_data_hybrid(graph_idx, 2, x_axis_idx, "[MB/s]", [line_val], mem_set)
    for i, graph in enumerate(gs):
        graph.plot(f_out, out_dir + "/")

# Another example is comparing the actual runtime with the ideal runtime.
graph_idx = 1
x_axis_idx = 0
mem_set = [16, 17]
for line_val in gen_plots._spaces:
    f_out = gen_plots.short_str[2] + "-" + line_val + "-" + \
            "RUNTIME" + "_vs_" + \
            gen_plots.short_str[x_axis_idx]
    gs = gen_plots._construct_graph_data_hybrid(graph_idx, 2, x_axis_idx, "[s]", [line_val], mem_set)
    for i, graph in enumerate(gs):
        graph.plot(f_out, out_dir + "/")

print "IDEAL BW TABLE"
print_table(IDEAL_BW)
print "VBW TABLE"
print_table(VBW)

# Print speed-up table for 1 to 100 layers.
print "SPEEDUP TABLE"
for p in mpi_procs:
    print "Processes: ", p
    for line_val in gen_plots._spaces:
        val = []
        val += [str(gen_plots._construct_speed_up_table(graph_idx, 2, x_axis_idx, all_layers, [line_val], [p]))]
        if line_val and (len(val) > 0 and val[0]):
            print " ".join(["$" + line_val.replace("x", "\\times ") + "$"] + [" & "] + val)

print "SPEEDUP TABLE USING UNORD TIMES"
# Print speed-up table for 1 to 100 layers.
for p in mpi_procs:
    print "Processes: ", p
    for line_val in gen_plots._spaces:
        val = []
        val += [str(gen_plots._construct_speed_up_table(graph_idx, 2, x_axis_idx, all_layers, [line_val], [p], time=4))]
        if line_val and (len(val) > 0 and val[0]):
            print " ".join(["$" + line_val.replace("x", "\\times ") + "$"] + [" & "] + val)

print "GFLOPS TABLE"
print_table(GFLOPS)

print "CUSTOM ADDS TO MULS TABLE"
print_table([A2M_RATIO, A2M_GFLOPS, GFLOPS, A2M_PERC_PEAK, VBW_PEAK], mpi_ps=[24], layer_list=[100])

print "CUSTOM VECT ADDS TO MULS TABLE"
print_table([A2M_RATIO, VECT_RATIO, VECT_A2M_PERC_PEAK, VBW_PEAK], mpi_ps=[24], layer_list=[100])

# Plot the standard graphs which compare the main measures across layers, processes and threads.
for graph_idx in [0, 1]:
  for line_idx in [2]:
    for x_axis_idx in [0, 1]:
      if graph_idx != x_axis_idx:
        for y_axis_idx in [3, 4, 8, 9, 10, 11, 12, 14]:
          # The IDXs have to b distinct.
          f_out = gen_plots.short_str[line_idx] + "-" + \
                  gen_plots.short_str[y_axis_idx] + "_vs_" + \
                  gen_plots.short_str[x_axis_idx]
          gs = gen_plots._construct_graph_data(graph_idx, line_idx, x_axis_idx, y_axis_idx)
          for i, graph in enumerate(gs):
              graph.plot(f_out, out_dir + "/")

gen_plots._output_to_roofline(out_dir + "/RHS_")
gen_plots._output_to_roofline(out_dir + "/RHS_", unord=True)
