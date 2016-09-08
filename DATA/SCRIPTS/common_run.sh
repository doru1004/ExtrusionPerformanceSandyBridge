#!/bin/bash

script="$1"
i="$2"
j="$3"
res="$4"
mesh="$5"
lhs="$6"
      
echo "1 $i $j"
echo "layers = $mesh" >> $res
echo "number of threads = 1 $i $j" >> $res
mpirun -n 1 likwid-pin -c 0 python $script $i $j $res $mesh $lhs

echo "6 $i $j"
echo "number of threads = 6 $i $j" >> $res
mpirun -n 1 likwid-pin -c 0 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 1 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 2 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 3 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 4 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 5 python $script $i $j $res $mesh $lhs

echo "12 $i $j"
echo "number of threads = 12 $i $j" >> $res
mpirun -n 1 likwid-pin -c 0 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 1 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 2 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 3 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 4 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 5 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 6 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 7 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 8 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 9 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 10 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 11 python $script $i $j $res $mesh $lhs 

echo "24 $i $j"
echo "number of threads = 24 $i $j" >> $res
mpirun -n 1 likwid-pin -c 0 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 1 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 2 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 3 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 4 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 5 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 6 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 7 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 8 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 9 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 10 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 11 python $script $i $j $res $mesh $lhs  : -n 1 likwid-pin -c 12 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 13 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 14 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 15 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 16 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 17 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 18 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 19 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 20 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 21 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 22 python $script $i $j $res $mesh $lhs : -n 1 likwid-pin -c 23 python $script $i $j $res $mesh $lhs 
