from firedrake import *
from pyop2 import op2
from pyop2 import hpc_profiling as p
from pyop2.configuration import configure
import sys
import random

spaces = [('CG', 1), ('CG', 2), ('CG', 3), ('DG', 0), ('DG', 1), ('DG', 2), ('DG', 3)]

times = 100

def get_prod_spaces(ind_list):
	prod_spaces = []
	for s1 in ind_list:
		for s2 in ind_list:
			prod_spaces.append(_fuse(spaces[s1], spaces[s2]))
	return prod_spaces

def _fuse(s1, s2):
	return s1[0] + str(s1[1]) + "x" + s2[0] + str(s2[1])

def mbw_additional_info(mesh, layers, V):
	# Compute the dofs per column of the horizontally discontinous space: dg_dpc
	# Replace fam with DG and compute the coordinate correction term: dg_coords
	# The correction term represents the memory footprint in byes of the number of degrees of freedom 
	# that the MBW is being overstimated by.
	# The correction term is applied once to the final MBW data volume computation.
	flat_elem = V.flattened_element.entity_dofs()
	entities = []
	for i in range(len(flat_elem)):
		entities.append(len(flat_elem[i]))
	dg_dpc = sum(V.dofs_per_column * entities)

	# Number of degrees of freedom in a DG1xCG1 column
	dg_coords_dpc = mesh.coordinates.cell_node_map().arity * (layers + 1)
	# number of cells in the horizontal
	# number of extra degrees of freedom per column
	# number of values for each degree of freedom (dim of the base mesh plus 1)
	# number of bytes
	dg_coords = mesh.num_cells() * (dg_dpc - dg_coords_dpc) * (mesh.coordinates.dat.dim[0] + 1) * mesh.coordinates.dat.dtype.itemsize
	return dg_dpc, dg_coords
