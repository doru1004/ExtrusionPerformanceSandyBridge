from firedrake import *
from pyop2 import op2
from copy import copy
from pyop2 import hpc_profiling as p
from pyop2.configuration import configure
import sys
import math
from pyop2.mpi import MPI
from common import spaces, times, mbw_additional_info

parameters['coffee']['compiler'] = "gnu"
parameters['coffee']['simd_isa'] = "avx"
parameters['coffee']['Ofast'] = True

print "Start building Mesh"
layers = int(sys.argv[4])
mesh = Mesh("../../MESHES/square_"+str(layers)+"_1.msh")

print "Finish building Mesh"
m = ExtrudedMesh(mesh, layers=layers, layer_height=0.1)
print "Finish building Extruded Mesh"

results = {}
filename = sys.argv[3]

ind1 = int(sys.argv[1])
ind2 = int(sys.argv[2])

fam = spaces[ind1][0]
deg = spaces[ind1][1]
vfam = spaces[ind2][0]
vdeg = spaces[ind2][1]

argl = len(sys.argv)
# RHS or LHS analysis
lhs = int(sys.argv[5]) if argl > 5 else 1

parameters['assembly_cache']['enabled'] = False

V = FunctionSpace(m, fam, deg, vfamily=vfam, vdegree=vdeg)

name = "%s%dx%s%d" % (fam, deg, vfam, vdeg)

dg_dpc, dg_coords = mbw_additional_info(mesh, layers, V)

t = TestFunction(V)

if lhs == 0:
	trial = TrialFunction(V)
	f_lhs = trial * t * dx
	F_lhs = assemble(f_lhs)
elif lhs == 1:
	f = t * dx
	F = assemble(f)
elif lhs == 2:
	f1 = Function(V)
	f1.assign(10.0)
	f = f1 * t * dx
	F = assemble(f)
elif lhs == 3:
	f1 = Function(V)
	f1.assign(10.0)
	f2 = Function(V)
	f2.assign(20.0)
	f = f1 * f2 * t * dx
	F = assemble(f)

MPI.comm.barrier()
with configure('hpc_profiling', True):
	with configure('region_name', ("LHS_" if lhs == 0 else "RHS_") + name):
		with configure('only_indirect_loops', True):
			with configure('papi_flops', True):
				with configure('dg_dpc', dg_dpc):
					with configure('dg_coords', dg_coords):
						p.reset()
						p.output_file(filename)
						with configure('times', times):
							with configure('iaca', True):
								if lhs == 0:
									print "Execute LHS assembly"
									F_lhs.M._force_evaluation()
									F_lhs = assemble(f_lhs)
								else:
									print "Execute RHS assembly"
									MPI.comm.barrier()
									F = assemble(f)

						with configure('randomize', True):
							with configure('times', times):
								if lhs == 0:
									for i in range(1):
										F_lhs.M._force_evaluation()
										F_lhs = assemble(f_lhs)
								else:
									for i in range(1):
										MPI.comm.barrier()
						 				F = assemble(f)
						p.summary()
