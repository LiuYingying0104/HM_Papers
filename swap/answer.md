[vDNN](vDNN.pdf) MICRO'16
transparent to programmer, allocation, placement, movement and relase of daya是由system architecture和runtime system决定的。
optimize feature extraction layer
keep track of the inter-layer dependencies in the form of a dataflow graph
在forward propogation中，根据dataflow graph确定前向计算中不用了就给它release
在backward propodation中，依旧是根据是否后面还要使用，不用了就release。如果选择offload,会有prefetch操作
这里的prefetching只考虑了在第几层，比较简单，而且单位好像是一整个所有的
Design上的challenge：
1. 如果operations也选最memory efficient，会导致GPU利用率没有到达最大
2. 如果用计算最快的，会导致OOM
Design1-static offload all layers of Xs
Design2-static offload all layers of Conv 
Design3-dynamic 需要profiling
1-先看看all offload能不能满足
2-再看看不offload能不能满足
3-oversubscribe的话 用fast+offload 
4-一层一层试试看，如果performance-efficient的没法满足，就改用memory-efficient的
用heuristic去找到offload和performance-effective之间operation的最佳解

其他：cudaMalloc和cudaFree会需要synchronous，所以用了memory pool的方法
扫了一眼代码（别人的implementation），感觉是一个建立在在cuDNN API之上的，新增了两个stream辅助memory management
cuda程序是知道neural network的信息的，会通过metaData传递信息
也有Bound，如果说每个offload都塞不下GPU那就没办法运行
测试的时候是直接用cuda
矛盾点在于cuDNN是不关心网络结构的，但是vDNN却需要全局地去了解整个网络结构，

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. feature extraction layers 是majority of memory usage **现实也是这样
   2. 需要对program进行profiling
2. 这些假设下论文的好处在哪里
   1. profiling的时间也不长
3. 好处主要体现在哪些项目的简化上
   1. 集成在cuDNN上，不需要额外的别的调用


[Chen Meng](Chen%20Meng.pdf)  In Proc. of ML Systems Workshop in NIPS. 2017
Graph-based optimization
design: allocate GPU memory budget to those with short life-cycle/ swap with long life-cycle
需要估计all nodes的completion-times (对应于DAG的weight)，用pre-run或者static analysis
Special design for seq2seq：移除attention score

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. Tensorflow1.0 static control flow graph
2. 这些假设下论文的好处在哪里
   1. without requring any changes to existing model descriptions
3. 好处主要体现在哪些项目的简化上


[DRAGON](DRAGON.pdf) SC'18
novel driver design: 修改了nvidia-uvm的driver
我的理解是GPU的数据在HM和NVM上都有replica，把GPU仅仅当作与cache。所以严格来说并不是swapping或者offload
三个API接口：
1-dragon_map() 主要负责将一个存储在 NVM（非易失性存储）上的文件映射到 GPU 的统一虚拟地址空间（Unified Virtual Address Space, UVA）
2-dragon_sync() 同步dirty page 到NVM
3-dragon_unmap() release resources
我的理解是cuda本身的UM是只map到CPU memory上，现在是map到NVM上，然后作两级的cache
本质上performance gain是来自于paging的readahead 

idea：如果在这个基础上去做一个prefetch呢？

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. 训练的数据或者模型非常非常大，在host都放不下
2. 这些假设下论文的好处在哪里
   1. 可以容纳得下，理论上没有upper bound
3. 好处主要体现在哪些项目的简化上
   1. 不需要human programming


[moDNN](moDNN.pdf) DATE'18
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[Layer-Centric](Layer-Centric.pdf) Archit. Code Optim 2018 
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[Tflms](TFLMS.pdf) 2018
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



[OC-DNN](OC-DNN.pdf) HiPC'19
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[Zhang Junzhe](Zhang%20Junzhe.pdf) 2019
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上




[GTBM](GTBM.pdf) HPDC'20
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



[AutoTM](AutoTM.pdf) ASPLOS'20
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[SwapAdvisor](SwapAdvisor.pdf) ASPLOS'20
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[Sentinel](Sentinel.pdf) HPCA'21
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[FlashNeuron](FlashNeuron.pdf) FAST 21
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[ZeRO-Offload](ZeRO-Offload.pdf) ATC'21
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



[ZeRO-Infinnity](ZeRO-Infinity.pdf) SC'21 （这个是ZeRO-Offload的延续）
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[DeepUM](DeepUM.pdf) ASPLOS'23
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



[G10](G10.pdf) MICRO'23
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上




[TMOF](TMOF.pdf) HPCA'23
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



[CachedArrays](CachedArrrays.pdf) IPDPS'24
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



[DeepTM](DeepTM.pdf)  IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS 2-24
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



这一个派别的主要发展过程、主要假设、主要理论依据、主要成果
这一派最适合什么时候使用、最不适合什么场合使用