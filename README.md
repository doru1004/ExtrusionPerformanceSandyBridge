# ExtrusionPerformance
Test harness for exploring the performance of finite element assembly in Firedrake.

# Dependencies

In order to run the tests a Firedrake installation is needed.

Clone the Firedrake repository referenced in the paper.

Please follow the installation instructions detailed in the Firedrake
repository:

docs/source/download.rst

The meshes used in the experiments have been archived separately.

In the main folder of the test harness repository create the following folder:

mkdir MESHES

Before running the experiments please un-archive the meshes into

MESHES/

# Running the tests

Set the environment and run test script:

. set.env

cd DATA/SCRIPTS

sh run.sh

# Output

The output for generating the results in the paper has been included.

MASS_FRHS_NEW_7MV_2/

The plots used in the paper are found in:

MASS_FRHS_NEW_7MV_2/ROOFLINE

