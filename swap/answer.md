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
本质上performance gain是来自于paging的readahead，因为跟它对比的直接显式地用fread/fwrite to stage data to/from NVMs.

idea：如果在这个基础上去做一个prefetch呢？

其他：UM-P和Hostreg的区别。两个都是可以在GPU上访问CPU的DRAM，区别是UM-P不需要programmar手动去分区，直接采用paging机制overscribe memory。

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. 训练的数据或者模型非常非常大，在host都放不下
2. 这些假设下论文的好处在哪里
   1. 可以容纳得下，理论上没有upper bound
   2. 不仅仅局限于DNN的优化，所有的GPU applications都可以
3. 好处主要体现在哪些项目的简化上
   1. 不需要human programming


[moDNN](moDNN.pdf) DATE'18
before training, allocate a memory space, record allocation offset
implement一个memory manager
有面对multi-GPU的特殊优化，就是weights的累计不是同步传输的，而是每个gpu执行完立马交给CPU来算
三个技术：
1-offloading and prefetching 区别是不是offload全部的或者全部conv
2-sub-batch
3-convolution algorithm selection 跟vDNN差不多,区别是不只是选择最快和最慢的

workflow：
a-要个graph
b-用profiling算出一个合适的batch-size
c-static scheduling 这里scheduling是直接作simulation，模拟整个task运行过程,offload是在allocate失败的时候发起的，traverse所有可能的offload策略，选个最少的
d-code generation

对于每个task的时间估计是需要profiling来确定的
为了避免fragmentation，是manage了一个memory pool，如果fragmentation太严重，就全部先offload/
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
      1. 需要一个TDFG （task and data flow graph）
      2. 保证从t时刻开始15%的memory是能塞进这个batch里的（所以相当于只是解的一个子集），有时候即使memory够了但是依然无法运行是因为fragmentation
2. 这些假设下论文的好处在哪里
   1. 相比于vDNN来说更加automatic一点
   2. algorithm更多种一点
3. 好处主要体现在哪些项目的简化上
   其实没简化，建立在vDNN之上，也是有两个cuda stream


[Layer-Centric](Layer-Centric.pdf) Archit. Code Optim 2018 
1-intra-layer memory reuse 改变梯度计算和激活层计算的顺序
2-inter-layer memory reuse and offload 固定一块大小的memory来反复reuse和offload
实现的时候是一个C++的库，分为三个部分：Tracer, IntraLayer和InterLayer
workflow:
在formal training之前先把ref都设置好
swapping就是如果前向用完了（ref count=0）那就把data offload到cpu, backprop的过程中提早一个layer把它弄回来
小的design：压缩数据
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. single GPU
   2. 把layer抽象为时间单位，提早一个layer prefetch
   3. 如果应用需要都改在特定的框架上，不generic
2. 这些假设下论文的好处在哪里
   1. 比较简单，无需考虑实际的执行时间
3. 好处主要体现在哪些项目的简化上
   1. 严格来说没有prefetching algorithm


[Tflms](TFLMS.pdf) 2018
是一个tensorflow中的module
方法：temporally sending "long lifetime" tensors in a GPU to a CPU and send back to GPU when necessary
这篇文章formal define了很多computation graph的操作
两个小优化：
1-考虑拓扑顺序，不需要频繁地swap in/out
2-如果有连续的，或者后面还会有依赖的，就先不swap out
computation graph构建的时候，并不知道tensor的actual size，所以没法衡量compute operation的cost
swap in的时间点是根据拓扑结构来的，提出了两种approach：direct order 和 chain-rule，direct order不考虑前向传播还是反向传播，就是提供upper和lower bound直接找出解（对于有多个解的random choose）
chain rule则考虑level和正反向传播

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. 使用tensorflow，有computation graph
2. 这些假设下论文的好处在哪里
   1. 直接formally 改写graph
3. 好处主要体现在哪些项目的简化上
   1. 不需要考虑cost，偏向于数学


[OC-DNN](OC-DNN.pdf) HiPC'19
用unified virtual memory的概念，把CPU 和 GPU 的memory都抽象成managed memory
implementation上是一个interception library，在CUDA runtime和硬件层之间，intercept memory相关的操作
有prefetch的策略，是使用cudaAPI给设备内存提供manage的建议
implement的时候是分成两个部分的 第一个部分是扩展Caffle的框架，把memory allocation的一些东西要改掉 还有内存拷贝的时候都抽象成unified memory（猜的）
第二个部分是interception library，主要是针对CUDA的runtime来给设备内存提供management的建议
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1.  the destination buffers are prefetched while the source buffers are most likely residing on the GPU memory already
   2.  single GPU setting
2. 这些假设下论文的好处在哪里
   1. 在这样的假设下，设计策略可以跳过一些复杂的情况
3. 好处主要体现在哪些项目的简化上
   1. prefetch策略并不是通过预测，只是提供一些hint


[Zhang Junzhe](Zhang%20Junzhe.pdf) 2019
把会用到的memory抽象成graph，vertices是memory blocks，edge表示lifetime有没有重叠，weight在vertice上表示memory block的大小
建模成graph coloring的问题，每个vertex都分配一种颜色，要求不能和neighbor撞色
提出了一种heuristic来找到graph coloring的optimal solution （先排序，然后再分配，best fit）
Autoswap:
1-小的不swap lifetime太短的不swap
2-根据那个时刻memory的load，lifetime以及大小算出来一个权重，用于计算swapping的priority
3-candidate用完就被swap out，提早swap in

关于implementation，抽象成一个device。应该也是根据前几个iteration来确定object的lifetime和size，之后可以开始构建smartpool
刚开始的时候是没有优化过的swap机制的，所以就是快溢出了就直接swap
operation index map到GPU分配的内存上

我有个疑问：每个iteration都是完全相同的吗？input没有大小区别吗 没有code没法看
不知道为什么没跟vDNN比

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. offline DSA, lifetime and size of all variables are known before allocating the first object
   2. 前几个iteration可以代表后面所有的情况
   3. 假设都使用memory-efficient的算法
2. 这些假设下论文的好处在哪里
   1. lifetime和size都知道了之后schedule应该会比较精准吧
3. 好处主要体现在哪些项目的简化上
   1. 不需要考虑conv layer本身算法workspace大小差异


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