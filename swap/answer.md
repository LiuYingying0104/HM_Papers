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

其他：cudaMalloc和cudaFree会需要synchronous，所以用了memory pool的方法

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. feature extraction layers 是majority of memory usage
   2. 需要对program进行profiling
   3. 用heuristic去找到offload和performance-effective之间operation的最佳解
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[Chen Meng](Chen%20Meng.pdf)  In Proc. of ML Systems Workshop in NIPS.
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[DRAGON](DRAGON.pdf) SC'18
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


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