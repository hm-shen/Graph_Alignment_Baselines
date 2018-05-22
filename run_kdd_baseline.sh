#!/bin/bash

init_tests()
{
    # ========== initial tests ==========
    python baseTests.py\
           -p ./data/kdd2018/arenas/arenas990-1/arenas_combined_edges.txt\
           -t ./data/kdd2018/arenas/arenas990-1/arenas_edges-mapping-permutation.txt\
           -a ./data/kdd2018/arenas/arenas990-1/attributes/attr1-2vals-prob0.000000\
           -m attributes\
           -n arenas900
}

# run_basetest()
# {
#     # run baseTest.py
#     python baseTests.py -p $1 -t $2 -m $3 -a $4 -n $5 -i $6
# }


run_struct_baseline()
{
    # ========== struct_baseline ==========
    NOISY_LIST=( "990" )
    GRAPH_LIST=( "arenas" "PPI" "citation" "fblarge" )
    VERSION_LIST=( "1" )
    DATA_ROOT_FOLDER=( "./data/kdd2018/" )
    DATA_NAME="_combined_edges.txt"
    ATTRIB_ROOT="attributes/"
    ATTRIB_LIST=( "attr1-2vals-prob0.050000" )
    ALIGN_DICT="_edges-mapping-permutation.txt"

    # run exp on each graph
    for graph in "${GRAPH_LIST[@]}"
    do
        for noise in "${NOISY_LIST[@]}"
        do
            for version in "${VERSION_LIST[@]}"
            do
                for attrib in "${ATTRIB_LIST}"
                do
                    echo "INFO:========== "\
                         "CURRENT GRAPH IS ${graph}-${version} "\
                         "WITH PCT NOISE ${noise}"\
                         "AND ATTRIB ${attrib}"\
                         " =========="

                        input_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${graph}${DATA_NAME}"
                        align_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${graph}${ALIGN_DICT}"
                        attrib_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${ATTRIB_ROOT}${attrib}"
                        echo "${attrib_path}"
                        mode="attributes"
                        graph_name="${graph}-${version}"
                        log_name="./logs/${graph}_v${version}_noise_${noise}_${attrib}.log"
                        # start running!
                        python baseTests.py\
                               -p ${input_path}\
                               -t ${align_path}\
                               -m ${mode}\
                               -a ${attrib_path}\
                               -n ${graph_name} > ${log_name}
                done
            done
        done
    done
}

run_struct_baseline
#init_tests