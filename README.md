# Install Torch7
I compiled Torch7 by using following steps:
1. git clone https://github.com/torch/distro.git /home/Tools/torch_cuda-7.5 --recursive
2. Modify path_to_nvcc=/usr/local/cuda-7.5/bin/nvcc in the file /home/Tools/torch_cuda-7.5/install.sh 
3. Make sure the path of /usr/local/cuda-7.5 in the file ~/.bashrc
Then running
1. cd /home/Tools/torch_cuda-7.5 ; ./clean.sh
2. rm -rf ./install
3. remove the torch-activate entry from your shell start-up script (~/.bashrc or ~/.profile)
4. bash install-deps
5. ./install.sh
6. ./test.sh 
7. set LD_LIBRARY_PATH & PATH
export PATH="/home/Tools/torch_cuda-7.5:/home/Tools/torch_cuda-7.5/bin:/home/Tools/torch_cuda-7.5/install:/home/Tools/torch_cuda-7.5/install/bin:/home/Tools/torch_cuda-7.5/install/share/lua/5.1:$PATH"
export LD_LIBRARY_PATH="/home/Tools/torch_cuda-7.5/lib:/home/Tools/torch_cuda-7.5/install/lib/lua/5.1:/home/Tools/torch_cuda-7.5/install/lib:$LD_LIBRARY_PATH"
. /home/Tools/torch_cuda-7.5/install/bin/torch-activate

Installation Reference: http://torch.ch/docs/getting-started.html  and https://github.com/torch/distro

# Install Twitter Packages
Please install the related Twitter packages at Distributed learning in Torch (https://blog.twitter.com/2016/distributed-learning-in-torch) before running.
First we git clone packages
1. git clone https://github.com/twitter/torch-distlearn
2. git clone https://github.com/twitter/torch-dataset
3. git clone https://github.com/twitter/torch-thrift
4. git clone https://github.com/twitter/torch-autograd
5. git clone https://github.com/twitter/torch-ipc

we can go to each folder and run commands
1. luarocks install autograd
2. luarocks install thrift
3. luarocks install dataset
4. luarocks install ipc
5. luarocks install distlearn

# Multi-GPUs
We can test run.sh and speech.lua by modifying input and ouput.
