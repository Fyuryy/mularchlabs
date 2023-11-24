#!/bin/bash
#SBATCH --chdir /home/gcharles/A3
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 36
#SBATCH --mem 10G

make order_asm
make

echo " -------------------------------------------------- "
echo Running order program twice positioning the two explicit threads on same socket and on different sockets.
echo " -------------------------------------------------- "
echo REMOTE - STARTING AT date

numactl -C 0,1,2 ./order
numactl -C 0,1,20 ./order

echo REMOTE - 
echo " "
echo " -------------------------------------------------- "
echo End of tests
echo " -------------------------------------------------- "