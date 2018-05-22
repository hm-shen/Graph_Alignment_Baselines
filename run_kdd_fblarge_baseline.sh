#!/bin/bash

init_tests()
{
    # ========== initial tests ==========
    python baseTests.py\
           -p ./data/kdd2018/structure_data/arenas/arenas995-1/arenas_combined_edges.txt\
           -t ./data/kdd2018/structure_data/arenas/arenas995-1/arenas_edges-mapping-permutation.txt\
           -m degree\
           -n arenas995-1
}

run_basetest()
{
    # run baseTest.py
    python baseTests.py -p $1 -t $2 -m $3 -n $4
}


run_struct_baseline()
{
    # ========== struct_baseline ==========
    NOISY_LIST=( "990" "980" "970" "960" "950" )
    # GRAPH_LIST=( "arenas" "PPI" "citation" )
    GRAPH_LIST=( "fblarge" )
    VERSION_LIST=( "1" )
    DATA_ROOT_FOLDER=( "./data/kdd2018_old/structure_data/" )
    DATA_NAME="_combined_edges.txt"
    ALIGN_DICT="_edges-mapping-permutation.txt"

    # run exp on each graph
    for graph in "${GRAPH_LIST[@]}"
    do
        for noise in "${NOISY_LIST[@]}"
        do
            for version in "${VERSION_LIST[@]}"
            do
                echo "INFO:========== "\
                     "CURRENT GRAPH IS ${graph}-${version} "\
                     "WITH PCT NOISE ${noise}"\
                     " =========="
                input_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${graph}${DATA_NAME}"
                align_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${graph}${ALIGN_DICT}"
                mode="degree"
                graph_name="${graph}-${version}"
                log_name="./logs/${graph}_v${version}_noise_${noise}_feb06.log"

                # start running!
                run_basetest ${input_path}\
                             ${align_path}\
                             ${mode}\
                             ${graph_name} > ${log_name}
            done
        done
    done
}

run_struct_baseline
