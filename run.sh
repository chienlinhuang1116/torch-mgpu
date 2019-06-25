# The main program
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
echo $CUDA_VISIBLE_DEVICES

# run 6 nodes
th speech.lua --numNodes 6 --nodeIndex 1 --batchSize 1 --cuda --gpu 1 &
th speech.lua --numNodes 6 --nodeIndex 2 --batchSize 1 --cuda --gpu 2 &
th speech.lua --numNodes 6 --nodeIndex 3 --batchSize 1 --cuda --gpu 3 &
th speech.lua --numNodes 6 --nodeIndex 4 --batchSize 1 --cuda --gpu 4 &
th speech.lua --numNodes 6 --nodeIndex 5 --batchSize 1 --cuda --gpu 5 &
th speech.lua --numNodes 6 --nodeIndex 6 --batchSize 1 --cuda --gpu 6 &

# wait for them all
wait
