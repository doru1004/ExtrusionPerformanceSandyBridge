#!/bin/bash

# Argument one should say whether some of the hardware systems are disabled or not: speedstep, prefetching
# Nothing disabled: ALL
# Disable Speed Step: DSS
# Disable HW Prefetch: DHP
# Disable Turbo Boost: DTB

vers="$1"
lhs="$2"
run="$3"

side="RHS"
sside="rhs"
rfl="v*dx"
vectorized="1"

if [ x"$lhs" = "x0" ];
    then
    side="LHS"
    sside="lhs"
    echo "Running LHS."
fi

if [ x"$lhs" = "x1" ];
    then
    side="RHS"
    sside="rhs"
    echo "Running RHS."
    vectorized="0"
fi

if [ x"$lhs" = "x2" ];
    then
    side="FRHS"
    sside="frhs"
    echo "Running F RHS."
    vectorized="0"
fi

if [ x"$lhs" = "x3" ];
    then
    side="FFRHS"
    sside="ffrhs"
    echo "Running FF RHS."
fi

base="${PWD}/../../"
scripts="${base}/DATA/SCRIPTS"
# Version must include the type of test and mesh size
gnuplot_dir="${base}/DATA/MASS_${side}_${vers}"

res="${gnuplot_dir}/results_foraker_${sside}_mpi_${PERF_KIT_VERSION}.txt"
script="${scripts}/rhs_cgdg_single.py"

roofline="${scripts}/rooflineExtruded.py"
rfl_dir="${gnuplot_dir}/ROOFLINE"
iaca_dir="${gnuplot_dir}/IACA"

dir_script="${scripts}/createDir.py"
dir_script_gentle="${scripts}/createDirGentle.py"
# Create the directory for the roofline data and plots
python $dir_script $rfl_dir
python $dir_script_gentle $iaca_dir

base_iaca="${iaca_dir}/iaca_"
export PYOP2_IACA_OUT_FILE="${base_iaca}"

if [ x"$run" = "xrun" ];
    then
    echo "Running the experiments."
    for i in 0 3 4
    do
        for j in 0 3 4
        do
    		for mesh in 1 2 4 10 30 50 100
     		do
                sh common_run.sh "$script" "$i" "$j" "$res" "$mesh" "$lhs"
            done
    	done
    done
    echo "end" >> $res
fi

greedy_reader="${base}/greedy_compact_reader.py"

python $greedy_reader $res $gnuplot_dir $side $vectorized $likwid_dir

cat "$gnuplot_dir"/RHS__*_1.txt > "${rfl_dir}/RHS_1.txt"
cat "$gnuplot_dir"/RHS__*_6.txt > "${rfl_dir}/RHS_6.txt"
cat "$gnuplot_dir"/RHS__*_12.txt > "${rfl_dir}/RHS_12.txt"
cat "$gnuplot_dir"/RHS__*_24.txt > "${rfl_dir}/RHS_24.txt"

python $roofline "${rfl_dir}/RHS_1.txt" "${side} - 1" 0 $vectorized
python $roofline "${rfl_dir}/RHS_6.txt" "${side} - 6" 0 $vectorized
python $roofline "${rfl_dir}/RHS_12.txt" "${side} - 12" 0 $vectorized
python $roofline "${rfl_dir}/RHS_24.txt" "${side} - 24" 0 $vectorized

cat "$gnuplot_dir"/RHS__*_1_u.txt > "${rfl_dir}/RHS_1_u.txt"
cat "$gnuplot_dir"/RHS__*_6_u.txt > "${rfl_dir}/RHS_6_u.txt"
cat "$gnuplot_dir"/RHS__*_12_u.txt > "${rfl_dir}/RHS_12_u.txt"
cat "$gnuplot_dir"/RHS__*_24_u.txt > "${rfl_dir}/RHS_24_u.txt"

python $roofline "${rfl_dir}/RHS_1_u.txt" "${side} - 1" 1 $vectorized
python $roofline "${rfl_dir}/RHS_6_u.txt" "${side} - 6" 1 $vectorized
python $roofline "${rfl_dir}/RHS_12_u.txt" "${side} - 12" 1 $vectorized
python $roofline "${rfl_dir}/RHS_24_u.txt" "${side} - 24" 1 $vectorized

cat "$gnuplot_dir"/RHS__DG1*_12.txt > "${rfl_dir}/RHS_DG1xX_12.txt"
cat "$gnuplot_dir"/RHS__CG1*_12.txt > "${rfl_dir}/RHS_CG1xX_12.txt"
cat "$gnuplot_dir"/RHS__DG0*_12.txt > "${rfl_dir}/RHS_DG0xX_12.txt"

python $roofline "${rfl_dir}/RHS_DG1xX_12.txt" "${side} - 12" 0 $vectorized
python $roofline "${rfl_dir}/RHS_CG1xX_12.txt" "${side} - 12" 0 $vectorized
python $roofline "${rfl_dir}/RHS_DG0xX_12.txt" "${side} - 12" 0 $vectorized

cat "$gnuplot_dir"/RHS__DG1*_24.txt > "${rfl_dir}/RHS_DG1xX_24.txt"
cat "$gnuplot_dir"/RHS__CG1*_24.txt > "${rfl_dir}/RHS_CG1xX_24.txt"
cat "$gnuplot_dir"/RHS__DG0*_24.txt > "${rfl_dir}/RHS_DG0xX_24.txt"

python $roofline "${rfl_dir}/RHS_DG1xX_24.txt" "${side} - 24" 0 $vectorized
python $roofline "${rfl_dir}/RHS_CG1xX_24.txt" "${side} - 24" 0 $vectorized
python $roofline "${rfl_dir}/RHS_DG0xX_24.txt" "${side} - 24" 0 $vectorized
