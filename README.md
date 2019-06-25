# Install Torch7
Please compile Torch7 by using following steps:
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
Please install the related Twitter packages at Distributed learning in Torch (https://blog.twitter.com/2016/distributed-learning-in-torch) before running. First we git clone packages
1. git clone https://github.com/twitter/torch-distlearn
2. git clone https://github.com/twitter/torch-dataset
3. git clone https://github.com/twitter/torch-thrift
4. git clone https://github.com/twitter/torch-autograd
5. git clone https://github.com/twitter/torch-ipc

Then, we can go to each folder and run commands
1. luarocks install autograd
2. luarocks install thrift
3. luarocks install dataset
4. luarocks install ipc
5. luarocks install distlearn

# Multi-GPUs
We can test run.sh and speech.lua by modifying input and ouput.

# Findings
1. Before running the distributed learning, please make sure turn ACS off. Please run lspci -vvv and make sure you get "ACSCtl: SrcValid-" instead of "ACSCtl: SrcValid+" for PLX PCI-e switch. There are some information about GPU communications:
> https://github.com/twitter/torch-ipc/issues/17

> http://www.supermicro.com/support/faqs/faq.cfm?faq=20732

> https://devtalk.nvidia.com/default/topic/883054/cuda-programming-and-performance/multi-gpu-peer-to-peer-access-failing-on-tesla-k80-/1

2. If you get "ACSCtl: SrcValid+" for the PCI bridge: PLX Technology, run "setpci -s bus#:slot#.func# f2a.w=0000" to disable ACSCtl on the PLX switch. Please run 3 steps:
> lspci | grep -i plx ,  …check bus#:slot#.func#

> sudo lspci -s 03:08.0 -vvvv | grep -i acs ,  …check ACSCtl: SrcValid+

> sudo setpci -s 03:08.0 f2a.w=0000 ,  …make ACSCtl: SrcValid-

3. We can check the setting of GPU cards and their topo matrix using the command of "nvidia-smi topo --matrix".
4. The activation function of Relu is better than Tanh. But, ReLu may fall into 0% accuracy with the unsuitable learning rate. There is no such problem when using Tanh.
5. We may use the command of "nvidia-smi --loop=10 > nividia.log" to reduce the happening of "Segmentation fault" in torch-distlearn.

# Reference
1. Microsoft: F. Seide et al., "1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs," Interspeech 2014.
2. Amazon: http://www.nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf
3. Dougal Maclaurin, David Duvenaud, Matt Johnson, "Autograd: Reverse-mode differentiation of native Python"
4. Twitter: https://blog.twitter.com/2016/distributed-learning-in-torch
5. Yu & Deng’s "Automatic Speech Recognition, A Deep Learning Approach"


UPDATE 16th March 2017, by Chien-Lin Huang https://sites.google.com/site/chiccoclhuang/
