#!/bin/bash

run_basetest()
{
    # run baseTest.py
    python baseTests.py -p $1 -t $2 -m $3 -a $4 -n $5 -i $6
}

run_attrib_baseline()
{
    # ========== struct_baseline ==========
    NOISY_LIST=( "990" )
    ATTRIB_NOISY_LIST=( "0.000000" "0.100000" "0.200000" "0.300000" )
    GRAPH_LIST=( "dblp" )
    VERSION_LIST=( "1" )
    DATA_ROOT_FOLDER=( "./data/kdd2018/attributes_data/" )
    ATTRIB_ROOT="attributes/"
    ATTRIB_LIST=( "attr1-29vals" )
    DATA_NAME="_combined_edges.txt"
    ALIGN_DICT="_edges-mapping-permutation.txt"

    # run exp on each graph
    for graph in "${GRAPH_LIST[@]}"
    do
        if [ "${graph}" == "dblp" ]
        then
            num_noise=30
        fi

        for noise in "${NOISY_LIST[@]}"
        do
            for version in "${VERSION_LIST[@]}"
            do
                for attrib in "${ATTRIB_LIST[@]}"
                do
                    for attrib_noise in "${ATTRIB_NOISY_LIST[@]}"
                    do
                        echo "INFO:========== "\
                             "CURRENT GRAPH IS ${graph}-${version} "\
                             "WITH PCT NOISE ${noise} "\
                             "AND ATTRIB ${attrib}, "\
                             "${attrib_noise}"\
                             " =========="

                        input_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${graph}${DATA_NAME}"
                        align_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${graph}${ALIGN_DICT}"
                        attrib_path="${DATA_ROOT_FOLDER}${graph}/${graph}${noise}-${version}/${ATTRIB_ROOT}${attrib}-prob${attrib_noise}"
                        mode="attributes"
                        graph_name="${graph}-${version}"
                        log_name="./logs/${graph}_v${version}_noise_${noise}_${attrib}-prob${attrib_noise}.log"

                        # start running!``
                        run_basetest ${input_path}\
                                     ${align_path}\
                                     ${mode}\
                                     ${attrib_path}\
                                     ${graph_name}\
                                     ${num_noise} > ${log_name}
                    done
                done
            done
        done
    done
}

run_attrib_baseline
