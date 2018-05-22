#!/bin/bash

# python baseTests.py\
#        -p ./data/blog995-1/blog_combined_edges.txt\
#        -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#        -a ./data/blog995-1/attributes/attr1-25vals-prob0\
#        -n blog995-1

# netalign: 99.13%

# python baseTests.py\
#        -p ./data/blog995-1/blog_combined_edges.txt\
#        -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#        -a ./data/blog995-1/attributes/attr5-2vals-prob0\
#        -n blog995-1

# netalign: 99.26%; final: 96.06%; isorank: 74.04%; klau: 99.64%

# python baseTests.py\
#        -p ./data/blog995-1/blog_combined_edges.txt\
#        -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#        -a ./data/blog995-1/attributes/attr1-2vals-prob0.200000\
#        -n blog995-1

# netalign: Not Enough Memory because of too much initial aignments.

# python baseTests.py\
#        -p ./data/blog995-1/blog_combined_edges.txt\
#        -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#        -a ./data/blog995-1/attributes/attr1-30vals-prob0\
#        -n blog995-1

# netalign: Not enough memory
# second try:
# netalign: 99.20%, 4282 sec;
# final: 96.01% 440 sec;
# isorank 70.87% 773 sec;
# klau 99.65% 4644 sec.

# python baseTests.py\
#        -p ./data/blog995-1/blog_combined_edges.txt\
#        -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#        -a ./data/blog995-1/attributes/attr1-2vals-prob0.200000\
#        -n blog995-1

# Not enough memeory for netalign, klau, isorank because of their
# implementation
# final: 0.48% 25.26 ????

# python baseTests.py\
#        -p ./data/blog995-1/blog_combined_edges.txt\
#        -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#        -a ./data/blog995-1/attributes/attr1-2vals-prob0\
#        -n blog995-1

# Does final still have a low score? final: 11.80%


# python baseTests.py\
#        -p ./data/blog995-1/blog_combined_edges.txt\
#        -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#        -a ./data/blog995-1/attributes/attr4-2vals-prob0\
#        -n blog995-1

# final 92.02% 470 secs

# attr_array=( # "./data/blog995-1/attributes/attr2-2vals-prob0"
#              # "./data/blog995-1/attributes/attr3-2vals-prob0"
#              # "./data/blog995-1/attributes/attr1-5vals-prob0"
#              # "./data/blog995-1/attributes/attr1-10vals-prob0"
#              # "./data/blog995-1/attributes/attr1-15vals-prob0"
#              # "./data/blog995-1/attributes/attr1-20vals-prob0"
#     # "./data/blog995-1/attributes/attr1-25vals-prob0"
#     "./data/blog995-1/attributes/attr4-2vals-prob0"
#     "./data/blog995-1/attributes/attr1-2vals-prob0.100000"
#     "./data/blog995-1/attributes/attr1-2vals-prob0.150000"
#            )

# attr1-2vals prob 0.1


# # run experiments for each kind of attributes.
# for attr in "${attr_array[@]}"
# do
#     echo "=============== INFO: CURRENT ATTRIBUTES IS ${attr}. ================"
#     python baseTests.py\
#            -p ./data/blog995-1/blog_combined_edges.txt\
#            -t ./data/blog995-1/blog_edges-mapping-permutation.txt\
#            -a ${attr}\
#            -n blog995-1
# done

# 2-2: final: 53.76%, 296.7 sec; klau, isorank, netalign oom
# 3-2: final: 81.11%, 354.4 sec; klau, isorank, netalign oom
# 1-5: final: 57.57%, 335.3 sec; oom
# 1-10: final: 84.4%, 390.2 sec; oom
# 1-15: final: 91.30%, 408.63 sec; netalign 98.27% 109955 sec; klau
# 99.35% 17882 sec; isorank 36.11% 5472 sec
# 1-20: final: 93.93%, 442.76 sec; netalign 98.48% 8926 sec; klau
# 99.53% 11971 sec; isorank 50.47% 3347.17
# 1-25: final: 95.76%, 542.45 sec netalign 99.13% 5088.96 sec; klau
# 99.58% 5690 sec; isorank 62.27% 1575.08 sec


attr_array=( "./data/dblp_trial1/attributes/attr1-29vals-prob0.050000"
             "./data/dblp_trial1/attributes/attr1-29vals-prob0.000000"
             "./data/dblp_trial1/attributes/attr1-29vals-prob0.100000"
             "./data/dblp_trial1/attributes/attr1-29vals-prob0.150000"
             "./data/dblp_trial1/attributes/attr1-29vals-prob0.200000" )


# run experiments for each kind of attributes.
for attr in "${attr_array[@]}"
do
    echo "=============== INFO: CURRENT ATTRIBUTES IS ${attr}. ================"
    python baseTests.py\
           -p ./data/dblp_trial1/dblp_combined_edges.txt \
           -t ./data/dblp_trial1/dblp_edges-mapping-permutation.txt \
           -a ${attr}\
           -n dblp_trial1
done

# final:
# noise level 0.05: 67.89% 508.44 sec
# noise level 0.00: 85.37% 509.16 sec
# noise level 0.10: 56.52% 660.08 sec
# noise level 0.15: 46.72% 507.95 sec
# noise level 0.20: 38.31% 508.72 sec
