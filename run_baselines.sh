#!/bin/bash

data_folder="./data/"
# graphs=( "karate" "arenas" "PPI" )
# ratios=( "0.0" "2.5" "5.0" "7.5" "10.0" )
graphs=( "arenas" )
ratios=( "0.0" )
msg="ver1"
nfeats="2"
mval="3"

for graph in "${graphs[@]}"
do
    for ratio in "${ratios[@]}"
    do
        echo "current graph is ${graph}"
        # create input arguments
        file_path=${data_folder}${graph}"/"${graph}"_comb_r"${ratio}"_"${msg}"_edges.txt"
        true_align=${data_folder}${graph}"/"${graph}"_align_r"${ratio}"_"${msg}"_permut.p"
        attribs=${data_folder}${graph}"/"${graph}"_comb_r"${ratio}"_f"${nfeats}"_m"${mval}"_"${msg}"_attr.p"
        name=${graph}
        # run baseline test
        python baseTests.py -p ${file_path} -t ${true_align} -a ${attribs} -n ${name}

    done
done
