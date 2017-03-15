local opt = lapp [[
Train a CNN classifier on CIFAR-10 using AllReduceSGD.
   --nodeIndex         (default 1)         node index
   --numNodes          (default 1)         num nodes spawned in parallel
   --batchSize         (default 32)        batch size, per node
   --learningRate      (default 0.01)      learning rate
   --cuda                                  use cuda
   --gpu               (default 1)         which gpu to use (only when using cuda)
]]

-- Requires
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpu)
end
-- luarocks install autograd
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local Dataset = require 'dataset.Dataset'

-- Build the AllReduce tree
local tree = require 'ipc.LocalhostTree'(opt.nodeIndex, opt.numNodes)
local allReduceSGD = require 'distlearn.AllReduceSGD'(tree)

-- Print only in instance 1!
if opt.nodeIndex > 1 then
   xlua.progress = function() end
   print = function() end
end

-- Adapt batch size, per node:
opt.batchSize = math.ceil(opt.batchSize / opt.numNodes)
print('Batch size: per node = ' .. opt.batchSize .. ', total = ' .. (opt.batchSize*opt.numNodes))

local isize = 680
local hsize = 2048
local bsize = 8192
local steps = 10
local osize = 17774
local osizei = 16765

local processor = function(res, opt, input)
   input:copy(torch.deserialize(res))
   return true
end

-- Dataset for training
local dataset = Dataset('/data/intermediate/am-train-01/fr-ca/chienh_2016-09-15_10.52.58/train_tutorial-dnn/trainset/train',{partition = opt.nodeIndex,partitions = opt.numNodes})
local getBatch, numBatches = dataset.sampledBatcher({
   samplerKind = 'linear',
   batchSize = opt.batchSize,
   inputDims = {bsize,681},
   poolSize = 1,
   cuda = true,
   processor = processor,
})

-- Dataset for developement
local dataset2 = Dataset('/data/intermediate/am-train-01/fr-ca/chienh_2016-09-15_10.52.58/train_tutorial-dnn/trainset/eval',{partition = 1,partitions = 1})
local getBatch2, numBatches2 = dataset2.sampledBatcher({
   samplerKind = 'linear',
   batchSize = 1,
   poolSize = 1,
   inputDims = {bsize,681},
   cuda = true,
   processor = processor,
})

torch.manualSeed(0)
-- for DNNs, we rely on efficient nn-provided primitives:
local linear,params,acts = {},{},{}
linear[1], params[1] = grad.nn.Linear(isize,hsize)
acts[1] = grad.nn.ReLU()
linear[2], params[2] = grad.nn.Linear(hsize,hsize)
acts[2] = grad.nn.ReLU()
linear[3], params[3] = grad.nn.Linear(hsize,60)
acts[3] = grad.nn.ReLU()
linear[4], params[4] = grad.nn.Linear(60,hsize)
acts[4] = grad.nn.ReLU()
linear[5], params[5] = grad.nn.Linear(hsize,hsize)
acts[5] = grad.nn.ReLU()
linear[6], params[6] = grad.nn.Linear(hsize,osize)

local lineari,paramsi,actsi = {},{},{}
lineari[1], paramsi[1] = grad.nn.Linear(isize,hsize)
actsi[1] = grad.nn.ReLU()
lineari[2], paramsi[2] = grad.nn.Linear(hsize,hsize)
actsi[2] = grad.nn.ReLU()
lineari[3], paramsi[3] = grad.nn.Linear(hsize,60)
actsi[3] = grad.nn.ReLU()
lineari[4], paramsi[4] = grad.nn.Linear(60,hsize)
actsi[4] = grad.nn.ReLU()
lineari[5], paramsi[5] = grad.nn.Linear(hsize,hsize)
actsi[5] = grad.nn.ReLU()
lineari[6], paramsi[6] = grad.nn.Linear(hsize,osizei)
-- Cast the parameters
params = grad.util.cast(params, opt.cuda and 'cuda' or 'float')

-- Make sure all the nodes have the same parameter values
allReduceSGD.synchronizeParameters(params)

-- Loss:
local logSoftMax = grad.nn.LogSoftMax()
local crossEntropy = grad.nn.ClassNLLCriterion()

-- Define our network
local function predict(params, input)
   local h1 = acts[1](linear[1](params[1], input)) 
   local h2 = acts[2](linear[2](params[2], h1))    
   local h3 = acts[3](linear[3](params[3], h2))    
   local h4 = acts[4](linear[4](params[4], h3))    
   local h5 = acts[5](linear[5](params[5], h4))    
   local h6 = linear[6](params[6], h5)             
   local out = logSoftMax(h6)
   return out
end

-- Define our loss function
local function f(params, input, target)
   local prediction = predict(params, input)
   local loss = crossEntropy(prediction, target)
   return loss, prediction
end

local df = grad(f, {optimize = true,stableGradients = true})

print('Train a neural network')

local lr = 0.04
local preAcc = 2.0
local msize = 512
local idx = bsize / msize
local labelAcc, correct, difAcc, curAcc

-- Load multilingual initial models
paramsi = torch.load("mx/efi256-f.2")
for s = 1, 5 do
   params[s] = paramsi[s]
end
--params = torch.load("clean256.1")

for s = 1, steps do
   local tm = torch.Timer()
   for t = 1, numBatches() do
      local batch = getBatch()
      local x = torch.reshape(batch.input[1],idx,msize,681)
      for f = 1, idx do
      local grads, loss, prediction = df(params,x[f]:sub(1,msize,1,680),torch.reshape(x[f]:sub(1,msize,681,681),msize))
      allReduceSGD.sumAndNormalizeGradients(grads)
      for layer in pairs(params) do
         for i in pairs(params[layer]) do
            params[layer][i]:add(-lr, grads[layer][i])
         end
      end
      end
      xlua.progress(t,numBatches())
   end
  
   allReduceSGD.synchronizeParameters(params)
  
   local sTime = string.format('%.2f', tm:time().real)
   correct = 0
   for t = 1,numBatches2() do
      local batch = getBatch2()
      local outputs = predict(params,batch.input[1]:sub(1,bsize,1,680))
      local vals, preds = outputs:max(2)
      match = preds:select(2, 1):eq(torch.reshape(batch.input[1]:sub(1,bsize,681,681),bsize):narrow(1, 1, preds:size(1)))
      correct = correct + torch.sum(match)
      xlua.progress(t,numBatches2())
   end
   curAcc = (correct/(numBatches2()* bsize))*100
   labelAcc = string.format("%.2f%%", curAcc)
   difAcc = curAcc - preAcc
   print('Epoch'..s..' LR'..lr..': Acc '..labelAcc..' Sec '..sTime)
   if curAcc < preAcc then break end
   preAcc = curAcc
   torch.save(string.format("dnn%d.%d",msize,s),params)
   if difAcc < 0.6 then lr = lr / 2.0 end
end

