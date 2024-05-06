input="/home/nx/ningxin/.data/graph/soc-pokec-renumber.eladjs"
input_binary="/home/nx/ningxin/.data/graph/soc-pokec.bcsr"
# -1 表示使用cpu
device="-1"

# BFS
echo "BFS with cpu"
./bfs-async --input $input --device $device

echo "BFS with gpu"
./bfs-async --input ${input_binary} --device 0

# PageRank
echo "PageRank with cpu"
./pr-async --input $input --device $device

echo "PageRank with gpu"
./pr-async --input ${input_binary} --device 0

# CC
echo "CC with cpu"
./cc-async --input $input --device $device

echo "CC with gpu"
./cc-async --input ${input_binary} --device 0

# SSSP

echo "SSSP with cpu"
./sssp-async --input $input --device $device

echo "SSSP with gpu"
./sssp-async --input ${input_binary} --device 0