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
Dymem host-side runtime interfaces with GPU. （这里说runtime不是说online scheduling）
workflow:
1-graph constructor
2-tensor scheduler 两个cuda stream: 跟vDNN类似 用完了就offload。
prefetch的策略：不早于两个conv prefetch
3-unified GPU memory pool (为了减少fragmentation)
把tensor分成三个类型：1-ref一次的 （low）2-临时的(high) 3-复用的 （motivation应该也是根据lifetime）

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. 应该是跟vDNN的框架类似，直接用cuda程序编写网络做测试？并不依赖于任何框架
   2. 考虑的网络种类不包含与RNN
2. 这些假设下论文的好处在哪里
   1. 简单的prefetech策略是有效的
3. 好处主要体现在哪些项目的简化上
   1. fragementation


[AutoTM](AutoTM.pdf) ASPLOS'20
workflow:
1-DNN model给nGraph，检测nodes，tensors，和使用的kernel
2-record kernel execution time
3-把profiling的信息和DAG都给memory optimier (Julia)
提出了两种assignment的方式：1-static 要么在PMM要么在DRAM 2-swapping
正式的定义和数学建模,用Gurobi求解

这个建模的过程当中并没有考虑到memory fragmentation，方法是先用schedule出来的结果跑一下，如果exceed了，那就降低memory budget重新schedule。

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. no data-dependent control behavior
   2. tensors are immutable
2. 这些假设下论文的好处在哪里
   1. 可以建模并求出解，因为是NP-complete
3. 好处主要体现在哪些项目的简化上


[SwapAdvisor](SwapAdvisor.pdf) ASPLOS'20
input: dataflow graph (topological order, mapping in memory pool, nums)
选一个合法的schedule和memory allocation作为初始值，提供给swapAdviser，然后加swap in/out的operator
然后用一个simulator去模拟overall execution time
用GA去提出新的plan，然后再模拟再优化 （用softmax去select children）crossover point randomly choose，mutate是基于已有的dataflow graph，遍历所有节点，有概率P发生变异
memory allocation也是类似的建模，设计了方法缩小搜索空间
swap-out: 移除未来最久不用的，用double scan来确定初始的时候GPU内会存在哪些tensor
swap-in: 在保证safe的前提下：complete a pair of swap out and swap-in as early as possible
Simulator是用了真实的operator的执行时间来算的
implementation: 用Python写了遗传算法和simulator，修改了MXNet
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. 知道每个kernel的执行时间
   2. 有dataflow graph
2. 这些假设下论文的好处在哪里
   1. simulator比较精确
3. 好处主要体现在哪些项目的简化上


[Sentinel](Sentinel.pdf) HPCA'21
profiling: one training step
OS-level: memory access at page level （通过poisoining PTE，写protection fault handler来implement）
Tensorflow-runtime: get tensor size and lifetime. / Add anotation to get DNN topology
一个page只能对应于一个tensor，把tensor和page中的位置对应起来，让OS知道page/tensor是属于哪个layer的
Design:
1-以layer为单位来做tensor management
2-把tensor分成short-lived和long-lived

Workflow：
1-dynamic profiling （integrate into Tensorflow）
2-reorganaize memory allocation 
   a-short live 根据layer allocate page
   b-long-lived 根据access的次数和layer分配位置, migration interval length是固定的，用GA或者Markov Chain MC来确定哪个interval是最好的。我觉得这个interval定义的有点抽象，应该就是个超参，来大概migration会花多长的时间。
focus on static graph不是因为需要静态分析之类的，是因为static graph的control flow是恒定的，在这个前提下training 的每个iteration其实都是相同的。
这个文章也兼容了dynamic graph和control dependency，如果有新的branch，那么重新profile
如何应用到GPU上？
1-用pinned memory，让GPU实际的访问是在CPU上完成的，这样可以profile
2-profile结束之后把那些tensor allocate到GPU上去，通过pointer switch.

implementation：
1-OS kernel for profiling （intercept protection fault）
2-Tensoflow Runtime 
   start_profile()
   end_profile()
   add_layer()用于标记layer之间间隙，提供hint给OS
三个thread：1.分类+算migration interval 2. swap in 3. swap out
GPU上的implementation也是遵循vDNN一样的范式，去create CUDA stream

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. the fast memory is at least larger than the peak memory consumption of those short-lived tensors.
   2. default version: static graph / CPU+Persistant memory
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上

comment: 说有很多small tensors with short lifetime
那如果第一个training step就已经OOM了呢？ 它一开始是基于哪个framework呢？ 会有memory的lower bound
本质上就是分类然后place，在一个固定的interval给它swap回来，我觉得这个固定的点不是很好，但文章中也解释了不用dynamic的原因



[FlashNeuron](FlashNeuron.pdf) FAST 21
是个library
总共分成3个部分：
1-allocation和deallocation 把要swap的分配在一边，不swap的分配在另一边，用P2P-DSA直接访问SSD
2-Offload Scheduler. 需要profiling iteration，记录的信息有1-tensor size/ 2-offload的time 
   3-compression ratio / 4-kernel execution time / 5-memory-resident object's total size
   step 1:先offload几个，直到能满足memory budget，如果没有delay正常的computation process 就直接结束
   step 2：如果找不到合适的，那就先用compression

关于implementation，看了代码是在pytorch的基础上做了拓展，所以文中提到的用Caffe用计算图算lifetime应该是没实现过。这里lifetime我有疑问，文章说通过reference counting mechenism去追踪lifetime，应该也是需要在profiling过程当中才知道的呀，那memory allocation的时候应该还不知道lifetime，是后面会再修改吗？

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. training iteration后面几个会和前面的保持一致
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


Zero系列的三篇文章：
https://basicv8vc.github.io/posts/zero
[ZeRO-Offload](ZeRO-Offload.pdf) ATC'21
是开源DeepSpeed PyTorch库的一部分
优化的对象主要是模型参数、梯度还有optimizer的动量方差，跟很多相关的work focus在activation layer上都不一样
https://www.deepspeed.ai/tutorials/zero-offload/ 
a-切割CPU和GPU的任务，前向和反向传播的计算量由于和batch size以及模型大小成正比，因此放在GPU，而范数计算和模型参数更新等则只和模型大小成正比，因此放在CPU.
b-并没有动态地swap in/out,就是规定哪些参数offload到哪里，哪些任务由CPU做，哪些由GPU做
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上
[ZeRO-Infinnity](ZeRO-Infinity.pdf) SC'21 （这个是ZeRO-Offload的延续）
同样是进行offload，ZeRO-Offload更侧重单卡场景，而ZeRO-Infinity侧重于工业场景
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上


[DeepUM](DeepUM.pdf) ASPLOS'23
由两个部分组成：1-DeepUM runtime / 2-DeepUM driver (Linux kernel module)
targets: pytorch
1-runtime:
wapper functions，把GPU的allocation都转化到UM的allocation上
所有会launch kernel的library function会加一个wrapper function
manage execution ID table，在launch kernel前把这个execution id发给driver
2-driver:
intercept page fault, provide to lookup correlation tables and update correlation tables'
Prefetching:
work at UM block level
两个correlation table: execution ID correlation table & UM block correlation table
用一个start pointer 和 end pointer来指示什么时候结束这个kernel的prefetching，这个kernel prefetching结束之后
去找下一个要执行的kernel，然后继续prefetch

会有pre-evict：least recently migrate / 未来不会被访问到的（这个很熟悉 之前有work是做这个的）
会invalidate UM blocks of inactive pytorch blocks（因为pytorch有自己的memory allocator）

comment：本质上是把历史都record下来，我觉得还是profiling啊……

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
   1. training iteration是一直在重复的
   2. 使用UM
2. 这些假设下论文的好处在哪里
   1. 没有boundary，理论上可以oversubscribe任意大小的model
3. 好处主要体现在哪些项目的简化上
   1. 不需要显式考虑tensor上的语义，只有block上的语义



[G10](G10.pdf) MICRO'23
三个部分组成：
1-tensor vitality analyzer (需要compiler) sizes & lifetime [offline compile-time profiling] 是一个静态分析工具
2-tensor migration scheduler （dynamic algorithm to find optimal solution）需要在修改code，inject instructions，修改的是GPU program
3-unified memory system （GPU, host and SSD are integrated into a unified space) 修改driver
用tensor size， storage bandwidth， host bandwidth去估计eviction和prefetch的时间

1-evict
对于evict的对象，用memory pressure减轻的程度和evict和prefetch的cost去算benefit-cost
优先考虑swap到SSD上，bandwidth满了才会考虑swap到Host上
iteratively search migration plan，直到满足memory budget

2-prefetch的时间点
由于evict之后GPU没那么满，G10就eagerly prefetch

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上

comment: 算的是tensor，但是migrate的时候却是以page为单位的诶


[TMOF](TMOF.pdf) HPCA'23
based on eager execution pytorch framework and CUDA library
在kernel execution的过程当中用Torch profiler obtain dynamic computational graph

1-decision engine:选择哪些tensor被swap出去 分为online和offline
a. online: 通过第一个iteration来确定reuse distance，从reuse distance最大的开始swap out，直到model size fit
b. offline: profiling run. 建模成MILP问题，跟autoTM很像，不一样的是把kernel和tensor access分离开来

2-channel contention avoidance
a. disjoint swapping (我觉得这是针对swap in来说的)
b. bidirectional overlapping （我看到里面的design是用一个token，避免同时swap out）

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上

comment：torch profiler也会有一定的overhead


[DeepTM](DeepTM.pdf)  IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS 2-24
Base on: Linux and Tensorflow
三个部分：
1-DeepTM Core
    Profiler: static computational graph
    Optimizer: formulate tensor management problem and provides optimization
    Allocator: allocate in HM 在每个内核开始时，将访问次数最多的张量分配到同一个页面
    根据重用距离（rk_t）升序排列张量，并将其分配到连续的内存页面上
    为了进一步优化内存带宽利用，DeepTM使用“连续迁移”策略，在迁移操作时将相邻的张量合并为一个事务进行迁移。
    Migrator: migrate in HM
    Aggregator: organize in page level
2-Heterogeneous Memory System
3-Kernel Computing

DRL:input 是以单个tensor为单位，包括它现在的位置，heat，和DRAM和PM中的left memory的多少，来决定action，是否migrate。根据migrate的成本，以及是否会造成OOM给它reward。
不断做出这样的action，知道training epoch结束，会给个总的评价

1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上



这一个派别的主要发展过程、主要假设、主要理论依据、主要成果
这一派最适合什么时候使用、最不适合什么场合使用