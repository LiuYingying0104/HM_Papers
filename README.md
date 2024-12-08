[cuDNN](All/cuDNN.pdf) 2014
optimized implementations for deep learning primitives
方法：C-language API for deep learning workloads

[vDNN](All/vDNN.pdf) MICRO'16
要解决的问题：模型太大 GPU内存不够 reduce memory usage of neural network  
提到的其他方法1：network pruning/quantization  
觉得他们不好的点：weights占比在整个memory usage中很小/loss of accuracy  
提到的其他方法2：page-migration based virtualization [51][52][34]  
觉得他们不好的点：underutilize PCIe bandwidth/performance overhead  
方法：要么直接release，要么offload to CPU Memory 之后再prefetch回来  

[Sublinear Cost](All/Sublinear%20Cost.pdf) 2016
要解决的问题：reduce memory consumption of DNN training
方法：用computation换memory, drop intermediate result. 针对于CNN和RNN
提到的其他方法1： swapping 
提到的其他方法2： parallel training
认为自己这一派好的原因：doesn't need additional communication over PCI-E/ save bandwidth
statement：size of parameters are relatively small comparing to size of the intermediate feature maps

[BPTT](All/BPTT.pdf) NIPS'16
要解决的问题：reduce memory consumption through backpropogation for RNN.
方法：dynamic programming
优点： tightly fit to almost any user-specified memory constraints
其实也是trade memory for computation time

[Thermostat](All/Thermostat.pdf) ASPLOS'17
要解决的问题：memory placement policy. 有新的storage technique，利用好two-tier memory system可以有net cost的improvement
其他的方法： naive policy（place pages into slow memory based on Access bits）
觉得它不好的地方：severe performance degradation
two-tier memory的两种做法：
1. paging mechanism 
2. migration mechanism 
假设：application transparent（critical for cloud computing environment）
方法：huge page aware/ place or migrate pages 
具体做法：continuously sample small fraction of pages, estimates page access rate by spatial extrapolation
用TLB miss的信息 as proxy for LLC misses/ correction mechanism.

[Bandana](All/Bandana.pdf) MLsys'18
要解决的问题：reduce DRAM footprint of embeddings；two-tier memory--DRAM & NVM
方法：
1. 把会一起读的store在一起；（我觉得是相当于把vector怎么分布成为一个block）
2. 用simulating来决定哪些存在DRAM里 （根据已经access的次数，选择一个合适的threshold）
文中的发现：如果要maximize bandwidth，需要4KB以上的read
假设：需要用到past access patterns of embedding vectors，针对的是facebook的recommandation DNN，没有普适性

[Superneurons](All/SuperNeurons.pdf) SIGPLAN'18
要解决的问题: GPU DRAM size limitation
方法: dynamic GPU memory scheduling runtime，把tensor作为unit，也有offload的部分
提到的其他方法1：static memory reduction （Caffe 和tensorflow内置的）
认为不好的点：不支持non-linear neural network，没有考虑memory variations
提到的其他方法2：split the network across GPUs (model parallelism)
认为不好的点：需要excessive communications--现在也有replica model的，但是也需要fit in GPU DRAM，没有利用NUMA架构
提到的其他方法3：MXNet，Tensorflow 用DAG去分析tensor的life span
提到的其他方法4：recompute
觉得不好的地方：没考虑dependency，只考虑了per-layer的cost，如果有复杂的依赖关系，可能会增加cost
提到的其他方法5：swap long-lived data from GPU to CPU, 没好好利用data communications
所有的：没有把memory和training speed平衡好

[Chen Meng](All/Chen%20Meng.pdf)  In Proc. of ML Systems Workshop in NIPS.
要解决的问题：model size too big for a single GPU
方法： dataflow-graph based/swapping 有特别针对于seq2seq的优化 / integrate into tensorflow 
提到的其他方法：model parallelism
认为不好的地方：bring communication overhead 
假设：feature maps占据主要内存消耗的部分
提到的其他方法2：Tensorflow重内置的best-fit with coalescing
认为不好的地方：没有memory optimization for training big models
提到的其他方法3：Unified Memory
认为不好的地方：sever performance loss
提到的其他方法4：offloading 
认为不好的地方：layer-wise CNN，不能用在sequence model上面
提到的其他方法5：re-computation （MXNet）
认为不好的地方：对于dynamic memory allocation，没法直接用

[moDNN](All/moDNN.pdf) DATE'18
要解决的问题：optimize memory usage in DNN training 
方法：修改DNN training code to match any given memory budget / offloading + prefetching /新的策略可以省memory
把4的几种结合起来
提到的其他方法1：network pruning -- loss accuracy，weights只占一小部分
提到的其他方法2：precision reduction -- loss accuracy，没有理论保证accuracy会被影响多少
提到的其他方法3：output re-computation -- longer training time
提到的其他方法4：static memory allocation/ batch partitoning/ out-of-core training
out-of-core的缺点：previous只有一个vDNN,太直接了
batch partitioning: 留给user的
static memory allocation的缺点（我的理解）：
1. data会根据live intervals被分到不一样的buffer中，但是每个时刻buffer不一定都是满的，但分配的时候算的是总和
个人觉得这里的缺点主要来自基于static allocation
2. long live interval的会一直占据资源不释放，占据了很多内存。个人觉得跟recompute的好处 或者offload的好处是对应的。

[Tflms](All/TFLMS.pdf) 2018
要解决的问题：GPU memory limitation for large DNN
方法：rewrite computational graph of a NN, swapping/ module in Tensorflow
其他方法1：reuse memory regions / compress a neural network/ use low precision
其他方法2：re-compute
其他方法3: use external memory such as CPU memory for temporarily storing intermediate results during training.
觉得3好的原因：可以训练更大的模型 / 可以apply到any neural network上
其他方法4：unified Memory (performance very pool)

[Gist](All/Gist.pdf) ISCA'18
要解决的问题：GPU main memory bottleneck
方法：DNN-layer-specific optimizations to reduce feature maps / encode representation and decode when use in backpropogation
其他方法1：reduce model size
认为不好的地方：only a small fraction of total memory footprint / lower precision reduce training accuracy
其他方法2：swapping
认为不好的地方：performance cost / use PCIe links 特别是对于distributed DNN training来说
其他方法3：reduce minibatch size
认为不好的地方：memory costly but power hungry
其他方法4：re-compute
认为不好的地方：long time re-compute

[DRAGON](All/DRAGON.pdf) SC'18
要解决的问题：dataset太大，需要用NVMe设备
方法：unified memory 用page fault 
提到的其他方法1：hardware -- hardware modification
提到的其他方法2：OS-level -- considerable overhead
提到的其他方法3：application-based -- limited to specific class of applications
UM--memory limited by host memory
提到的其他方法4： 写了一个custom NVMe driver, 可以直接在GPU喝NVM之间用RDMA传输数据，但是有内存限制（global memory）

在related work中的分类：
1. compiler-based techniques: anlyze data flow, inject code to automatically partition parallel loops and tasks into smaller region, and map them into CPU and GPU 
2. software-based approaches: user-level APIs along with runtimes 缺点：调用API时比较麻烦
3. OS and hardware-level: memory management with hardware modifications
   
[Layer-Centric](All/Layer-Centric.pdf) Archit. Code Optim 2018 (bad writing**)
要解决的问题：running an extreme-scale model in a signgle GPU 
方法：runtime data placement strategy, memory reuse
其他方法1：model quantization and compression/ network pruning
问题：performance degradation
其他方法2：CPU-GPU transfer
其他方法3：computation graph analysis

其他方法1：partition and distribute the modesl over multiple GPUs
认为不好的地方：not cost-effective due to excessive communication / guarantee convergence 
其他方法2：memory pool （store intermediate data between GPU and CPU memory）
其他方法3： extra computation 

把layer分成几个类别：
1. Feature extraction layer
2. Activation function layer (sigmoid ReLU)
3. Loss function layer 
4. Regularization function layer (dropout / batch normalization)

[Zhang Junzhe](All/Zhang%20Junzhe.pdf) 2019
要解决的问题：GPU limited memory size
其他方法1: model compression
问题：degrade model accuracy
其他方法2: memory swapping
问题：require manual intervention
其他方法3: data partition
问题：适用于data access是regular的情况
其他方法4：memory Sharing
可以分为两种，一种是in-place operation，另一种是把lifetime不重合的东西存在一个memory block上，这个需要entire computation Graph
其他方法5：recompute
问题：也是需要computation graph
方法：1.optimize memory allocation based on lifetime of all variables 2. swapping 

其他知识：memory pool (take over memory management from OS)
paging on GPU 需要改硬件/driver 


[Nimble](All/Nimble.pdf) ASPLOS'19
要解决的问题：page migration for tiered memory system

[Layup](All/Layup.pdf) TACO'19 (这篇写得挺细的)
要解决的问题：GPU memory limitation
其他方法1：model parallel (divide-and-conquer strategy)
问题：introduce much more complexity into the construction of distriburted models 
其他方法2：reduce memory consumption (CPU-GPU transfer / re-compute)
问题：layer type affacts execution time / no consideration of worspace data
其他方法3： model compression（pruning/ quantification）适用于embedded devices 
觉得superneuron不好的地方：does not consider the difference in execution time between the CPU-GPU transfer and the extra forward computation for different layer types.
假设：基于single GPU 
方法：multi-type data reuse 

[OC-DNN](All/OC-DNN.pdf) HiPC'19 (有点没看懂)
要解决的问题：out-of-core DNN training, GPU memory limitation
vDNN的limitation：需要manual effort
Unified memory：time performance
假设：single GPU
方法：develop UM primitives.

[CheckMate](All/checkmate.pdf) 2019
Automatic rematerialize large neural network optimally (应该就是re-compute)
goal: fit in an arbitrary network with memory budget while incurring the minimal additional runtime penalty from recomputation
其他方法1：checkpointing and rematerialization
其他方法2：reversible networks
其他方法3：distributed computation
其他方法4：activation compression (reduce accuracy)
假设：需要computation graph 

[Buddy Compression](All/Buddy%20Compression.pdf) International Symposium on Computer Architecture, 2020
Goal: increase effective GPU memory capacity and bandwidth.
方法：memory compression 采用一种分级压缩策略，按数据的重要性和冗余性分层处理，动态调整压缩比。
显存中的数据被分为多个小块（例如 128 字节块），每个块单独进行压缩和解压缩。
根据压缩效率和内存压力，动态选择是否启用压缩。

IBM Large model support: 
1. https://github.com/IBM/ 
2. https://www.ibm.com/docs/en/wmlce/1.6.0?topic=gsmf-getting-started-tensorflow-large-model-support-tflms-v2

[KARMA](All/KARMA.pdf) SC'20
其他方法1：model parallelism （viable approach to...）
问题：significant modifictaion of source code / complex, intrusive and enforce a non-trival bound on minimum number of GPUs for large models
其他的方法2：out-of-core (swapping)
问题：no efficient prefetching/ 同步的开销等  会带来performance overhead 
其他方法3：recompute
问题：有bound / could not be used to improve performance in distributed training / swap 放在distribute上面会有冲突
方法： layer swapping+redundant recomputing / support multi-GPU 

[AntMan](All/Antman.pdf) OSDI'20
schedule deep learning jobs on large-scale GPUs 

[DeepSpeed](All/DeepSpeed.pdf) KDD'20
tutorial: fastest BERT training

[GTBM](All/GTBM.pdf) HPDC'20
要解决的问题：GPU memory limitation
属于的派别：offloading and prefetching feature maps
其他方法1：reduce model size
问题：loss in training accuracy / parameter weights only account for a small frection
其他方法2：memory compression / data encoding 
问题：high performance overhead 
其他方法3：swapping
问题：1-vDNN和superneurons的缺点：不适用于non-linear networks 
2-memory fragmentation problem （which has different data size, varied resident duration, and dynamic reference counts, interleave with layers which have simple dependencies)

[Sage](All/Sage.pdf) VLDB'20
parallel graph analytics

[AutoTM](All/AutoTM.pdf) ASPLOS'20
要解决的问题：模型太大，DRAM不够
提到的其他方法1：用SSD做backing storage [13][14]
提到的其他方法2：memory management between CPU and GPU [8][44]
方法：提出一个framework，automatically move data between heterogenous memory devices. (compiler-based)
假设条件：computational graph is static (no data-dependent control behavior)
别的文章觉得它不好的点：kernel computation and its tensor accesses are coupled as one execution unit. 
Tensor-swap operation cannot be decoupled from its kernel, thereby losing an optimization opportunity of overlaping tensor swapping with an independent kernel's execution.

[SwapAdvisor](All/SwapAdvisor.pdf) ASPLOS'20
要解决的问题：模型太大，GPU memory不够，existing work不够general，无法适应所有的models
提到的其他方法1：lower-precision floating point/compression model parameter/quantization
觉得他们不好的点：affect model accuracy/需要heavy hyper-parameter tuning
提到的其他方法2：扔掉intermediate data [2][11][29]
觉得他们不好的点：cannot support large models（因为model parameter不方便re-compute）
自己这一类swapping的方法：在DNN computation的时候swap tensor between GPU and CPU [26][30][40][49]
觉得自己这一派好的点: 1. CPU memory比GPU memory更大更便宜/ 2.GPU 可以边通信边计算/ 3. GPU 和 CPU communication bandwidth 可以满足
这一派中其他方法不好的点：DNN computation structure是固定的，他们要套么没用这个信息，要么就是用的很基础
比如说[26][40]只swap activation，大模型还是不能fit
方法：genetic algorithm， predict what and when to swap precisely/ 
joint optimization over operator scheduling, memory allocation and swapping
假设条件：static dataflow graph with no control-flow primitives / single GPU
其他：文章把model的memory consumption分成三个类别：model parameters/intermediate results/scratch space

[Capuchin](All/Capuchin.pdf) ASPLOS'20
要解决的问题：模型太大，GPU on-board memory不够
提到的方法1：swapping 用CPU DRAM作为external memory
提到的方法2：recomputing 需要用到的时候再recompute
提到的方法3：use low precision computational
觉得它不好的地方：很难分析最后是怎么影响training accuracy的
基于1-2已有的works：layer-wise GPU memory management
觉得它不好的地方：太high-level了，coarse-grain，对computation graph进行static analysis[5][24][31][6]
硬件和输入大小会影响整个prediction / 缺乏量化数据 / pytorch和tensorflow的eager mode 没有computation graph
方法：make memory management decision based on dynamic tensor access pattern tracked runtime/ tensor-based
做三个事情：prefetch/evict/recompute
基于的假设/obsevation：1. 在training iteration的时候access patterns to tensor是固定的
1. major memory footprint in DNN training 是来自于intermediate layer outputs [13][24]
2. 不基于computation graph （适用于imperative programming environment）
其他：文章把model的memory consumption分成三个类别：feature maps / gradient maps/ convolution workspace （model parameters占比很小）
Programming中有两个mode：1. eager mode（对应于imperative programming），不生成计算图，但是因为没有optimization/需要interpret python code，所以overhead大 （例子：PyTorch/Tensorflow2.0）
1. graph mode（对应于declarative programming）before execution，computation graph is built （例子：TensorFlow1.0）
假设：checkpointing scheme依赖于static model architecture (or infer from an initial profiling batch)

[Sentinel](All/Sentinel.pdf) HPCA'21
要解决的问题：现在Heterogeneous Memory 越来越流行，用HM实现DNN training achieve larger memory capacity
提到的其他方法1：用 CPU-side system memory来扩容[5-11] / 用persistent memory来扩CPU的容 
based on DNN topology[7-10] / detailed domain knowledge [5][6][11]
觉得不好的地方：semantic gap between OS and memory management in deep learning framework (比如 lifetime不同可以放的位置不同/OS是page-level，而predict的是tensor-level，会造成fragmentation)[12][14]
不考虑lifetime的话，会把一些short-live的给挪动了[15]/hardware[16-18]
方法：dynamic profiling / OS / runtime-level profiling / performance model (让migration跟DNN training同时进行) 
其他：面向的是tensorflow

[ZeRO-Offload](All/ZeRO-Offload.pdf) ATC'21
要解决的问题：large model training
提到的其他方法1：(scale-out training) pipeline parallelism/ model parallelism/ ZeRO [5,28,7,10,21]
觉得不好的地方： having enough GPU devices （总量还是要够的）
提到的其他方法2：Heterogenous DL training [8,9,11,17,23,24,32-34] [18]
三类：
1. recompute [4]
2. compression [16]
3. externel memory [8,9,11,17,23,24,33] ***
觉得（现有的工作）不好的地方：1. 基本上target activation memory，没考虑model states(也就是静态数据) 2. 只考虑了CPU memory资源，没考虑CPU计算资源 3. 没考虑multiple GPU的情况
方法：主要是offload一些计算和memory到CPU中，集成到了pytorch的库当中，并支持多个GPU并行
其他：把内存分成了model states和residual states。我觉得可以理解为静态和动态

[Zico](All/Zico.pdf) ATC'21
要解决的问题：reduce system-wide memory consumption for concurrent training （GPU Sharing）
提到的方法派别1：temporally multiplexing [26,43,44]
觉得这个派别不好的地方：大多数时候计算资源没有被充分利用
提到的方法派别2：spatially sharing [10]
觉得这个派别不好的地方：working set太大无法fit in GPU memory --> 所以对于这个派别来说，需要reduce memory footprint
方法： forward时候memory usage在增加，backward时候memory usage在减少，因此可以共享一个memory pool，应该是用一个GPU kernel去monitor memory allocate和release的event，然后再用scheduler去schedule
其他：面向的是tensorflow

[ZeRO-Infinnity](All/ZeRO-Infinity.pdf) SC'21 （这个是ZeRO-Offload的延续）
要解决的问题：模型太大 GPU内存不够
1. How do we support the next 1000x growth in model size, going from models like GPT-3 with 175 billion parameters to models with hundreds of trillions of parameters?
2. How can we make large models of today accessible to more data scientists who don’t have access to hundreds to GPUs？
3. Can we make large model training easier by eliminating the need for model refactoring and multiple forms of parallelism?
方法：在heterogeneous memory上（GPU,CPU,DRAM）上allow model scale without requiring model code refactoring / memory-centric tiling 解决单层过大的问题/ bandwidth-centric partitioning 根据device带宽来决定分区/ overlap-centric design (把offload跟communication尽量并行运算)

[Xupeng](All/Xupeng.pdf) SIGMOD'21
要解决的问题：All-reduce architecture in heterogeneous environment. (distributed ML)

[Oliver](All/Oliver.pdf) NIPS'21 （formal definition）
要解决的问题：save memory during the training phase of DNN.
其他方法1：parallelism-based memory optimization
可以分为data parallism & model parallelism （前者distribute batch，后者distribute model）
weights replica / collective communications
其他方法2：rematerialization: trade memory for computation time
其他方法3：offloading: trade memory for data movements.
方法：combination of rematerialization and activation offloading.
假设：single GPU

[FlashNeuron](All/FlashNeuron.pdf) FAST 21
要解决的问题：memory capacity wall
方法：utilize SSD as training backing store. Direct communication between SSD and GPU utilizing GPUDirect.
其他方法：Use of multiple GPUs
问题： throughput--near linear/ cost--linear -> result: sub-optimal cost efficienct
其他方法2： utilize host CPU memory as backing store 
问题：training process on GPU contends with applications running on the CPU for memory bandwidth and capacity.


[KLOCs](All/KLOCs.pdf) ASPLOS'21
Heterogenous memory management
key idea: 当前的操作系统主要关注应用数据的分层管理，而忽略了内核对象（如文件系统元数据、网络缓存等）的高效管理

[PET](All/PET.pdf) OSDI'21
要解决的问题：optimize tensor programs 
用program transformation (fully equivalant / partially equivalant)
提出的方法：用partially equvalant + correction kernel

[DTR](All/DTR.pdf) ICLR'21
问题：GPU memory limitation
其他方法：static planning (assume static dataflow graph) 
其他方法2：Memory manager 
方法：greedy algorithm / make no assumptions about model's structure 
假设： linear feedforward？

[Bergman](All/Bergman.pdf) SIGPLAN'22
要解决的问题：Efficient access to disaggregate memory
用swap+cache-line access的混合模式，把冷数据依旧放在remote memory，热数据swap回来

[FlexHM](All/FlexHM.pdf) TACO'22
two-level NUMA design / periodical interaction between memory performance monitoring / place hot/cold pages

[POET](All/POET.pdf) ICML'22 (open-soiurce)
Enable training large neural networks on memory-scarce battery-operated edge devices. (constrains on memory and runtime)
(why training on edge? privacy)
其他方法：Paging to auxilaiary memory & rematerialization 
问题：significant increase in total energy consumption
paging is energy-intensive and often less efficient than rematerialization 
方法：use integer linear program to find energy-optimal solution / combine paging and rematerialization 

[Unity](All/Unity.pdf) OSDI'22
Goal: accelerate DNN training theough joint optimization of algebraic transformation and parallelization
(虽然不是特别相关 但是内容可以读)

[MoNet](All/MoNet.pdf) ICLR'22
Goal: automatic framework that minimize both memory footprint and computational overhead of DNN.
其他方法1：operator-level implementation changes
hand-craft techniques, no one-size-fit-all recipe, different implementations perform best on different architecture.
其他方法2：global, graph-level optimizations (checkpointing)
方法：local+global

[TSPLIT](All/TSPLIT.pdf) ICDE'22
要解决的问题：DNN太大， device memory不够
提到的方法1：distributed system [20-22]
觉得不好的点：bring high communication costs/ increase system complexity
提到的方法2：model compression
觉得不好的点：affects final accuracy of the model/ require heavy hyper-parameter tuning
提到的方法3：recompute or swap [27][19][17] tensor-wise [33-36] [37-38]
觉得不好的点：在某些operations下，依旧没法trainable / 大tensor的交换很time-consuming
方法：break operation boundary （micro-tensor）/ computation graph profile
假设：应该也是基于静态的computation graph的

[TMOF](All/TMOF.pdf) HPCA'23
要解决的问题：用host memory作为external memory来swap tensor，但是在data-parallel training system中效果不佳。
提到的其他方法1： swapping-based techniques [16,19,20,36,41,43,47]
其中[43]和[47]被认为是layer-wise （static computational graph）
其他都是tensor-wise：
Graph execution ML framework (tensorflow1.0) swapping 是offline的 [16,20]
Eager execution ML framework: [19,36] 也是预先定义的，说19是超出了capacity都swap，36是在第一个training iteration的时候profile
认为他们不好的点：design for single-GPU systems 如果apply到data-parallel system中的话，会引起PCIe channel的争用
方法：online和offline都有。online的部分跟[36]很像，swap-out reuse distance比较大的；offline 用dynamic profiling
contention-avoidance techniques: disjoint swapping (select disjoint sets of swapped-out tensors for different GPU nodes) + bidirectional overlapping (sheduele swap out and swap-in together)
假设：DNN，面向pytorch。感觉它的performance gain基本上来自于contention-avoidance
其他：关于distributed model training--BSP(bulk synchronous parallelism) 每个GPU有自己的gradient set，然后每n个training iterations之后gradient会累计

[DeepUM](All/DeepUM.pdf) ASPLOS'23
要解决的问题：用CUDA Unified Memory来解决GPU memory不够的问题
提到的其他方法1：data compression [6, 10, 18, 26, 34], 
提到的其他方法2： mixed-precision arithmetic [11, 17, 28]
提到的其他方法3： data recomputation [8, 16, 55]
提到的其他方法4： memory swapping [5, 6, 21, 24, 33, 45, 49–51, 55]
这一个派别下有两个类：
a. CUDA Unified Memory with page prefetching [5,35] 
这个类别下工作比较少的原因：address translation和page fault handling的overhead比较大 （相当于是disadvantages） 因为Page Fault发生的时候TLB会被lock住，可以用cudaMemPrefetchAsync或者cudaMemAdvise
好处：can handle larger DNN models / memory fragmentation
b. pure GPU memory swapping-in/swapping out [6, 21, 24, 33, 45, 49–51, 55].
假设：single GPU/ DNN，面向pytorch / CUDA / kernel execution patterns and their memory access patterns are mostly fixed and repeated in the DNN training workload.
方法：1. correlation preferching 2. page pre-eviction 2. page invalidation in GPU memory (remove unnecessary memory traffic betwwen CPU and GPU)
其他: 关于unified memory--CPU和GPU之间share single address space （也有其他名字，比如Virtual Memory, OpenCL, Intel oneAPI）
换句话来说，single memory address space that can be accessed by both CPU and GPU / GPU 处理page fault是以UM Block为单位的（512 Pages)
提出UM的motivation是GPU thread不能直接access CPU的memory space，所以需要programmer manually move。move的过程中会有PCIe上面overhead的问题

[G10](All/G10.pdf) MICRO'23
要解决的问题：hard to meet the scalability requirement of deep learning workloads
提到的其他方法1：expand limited GPU memory with flash memory. [67,68,61,19,21] 这些感觉是偏向technique方面的，bring Flash closer to GPUs
觉得他们不好的点：limited bandwidth by the PCIe interface
提到的其他方法2：提出特殊的针对DNN的方法 move data across heterogeneous memories [12,25,27,34]
觉得他们不好的点: complicates GPU memory management, hurt development productivity
方法：integrate host memory, GPU memory and flash memory into a unified memory space / 用compiler technique to characterize the tensor behaviors
看了代码，输入是转化过的code，类似于DNN网络结构信息，然后用静态分析等方法来计算
其他：因为需要融合了SSD，所以需要修改driver。但是NVIDIA driver不开源，因此需要用到simulator 
假设：single GPU，测试的是传统DNN和transformer

[HOME](All/HOME.pdf) IEEE Transaction on Computers 23
要解决的问题：GPU Memory not enough  
提到的其他方法1：data compression
觉得它不好的地方：loss accuracy / incur significant time overhead
提到的其他方法2：Data swapping [14,15,16,23]
提到的其他方法3：Data Recomputation [17]
提到的其他方法4：swapping+recomputation [18,19]
觉得它不好的地方：only consider partial DNN model information,18只局限在一部分种类的DNN，19的dataflow从第一层DNN来
关于选择search algorithm:ILP复杂度太大，SA,BO,RL,MC容易陷入local optimal，并且没法parallel searching，所以用PSO
方法：(novelty) consider holistic DNN model information; 三个action选一个--swapping recomputation retaining
假设：面向pytorch; 没有在real hardware platform上部署，用了一个time-cost model来predict大概需要的时间

[Occamy](All/Occamy.pdf) DAC'23
要解决的问题：reduce memory usage of DNN without affecting accuracy
其他方法1：memory offloading [3,4,5,6]
其他方法2：recomputation[4,6]
其他方法3：tensor decomposition[6]
其他方法4：model compression
认为他们不好的地方：limit to DNN training or 需要programmer change model
另一个类别（把memory allocation考虑进去的）：pytorch是在一开始就allocate所有的tensor，但忽略掉了memory size
/ tensorflow Lite Micro和TASO 使用了reuse的策略，但会造成fragmentation的问题 [1,8,29,30,31]
方法：设计了一个DNN compiler，感觉主要是在考虑memory allocate和deallocate上的优化；基于onnx-mlir compiler 
跟之前的比没有runtime overhead。
假设: DNN, 面向Pytorch

[CachedArrays](All/CachedArrrays.pdf) IPDPS'24
要解决的问题：data management in heterogeneous memory systems
Algorithm: 7
Application: 9, 13 (11,12,10,37) manually determine the reuse pattern and figure out a suitable data movement scheme to change the algorithm
Compiler: 8, 3, 14, 15 scalability and generalization problems
OS: 16-26 don't take into account future information about the data use and semantic information from the Application

[MAGIS](All/MAGIS.pdf) ASPLOS'24
要解决的问题：memory optimization for DNN
第一类：graph scheduling
其他的方法1：rematerialization [5,10,17,18,24,27–29,37,38,47]
其他的方法2：swapping [5,20,22,30,37–39,41,42,57]
其他的方法3: re-ordeing [3,22,58,72]
认为他们不好的地方：overhead太大；没有改变tensor的形状，limit potential optimization space
第二类：graph transformation [25,26,54,56,62] rule-based sub-graph substitution
1. Aggregation transformation
2. Interim transformation
3. Fission transformation 
认为他们不好的地方：complexity太大；trade-off: latency和memory


[GMT](All/GMT.pdf) ASPLOS'24
要解决的问题：application working sets grow，GPU需要access much larger data capacities
一类方法：UVM 问题：没有扩展到SSD
一类方法：扩展到了SSD的 问题：没有绕过CPU的控制，不能直接访问SSD

[DeepTM](All/DeepTM.pdf)  IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS 2-24
要解决的问题：：DNN training tensor management on heterogeneous memory
Transform performance optimization problem into ILP.
认为现有的swapping strategy的缺点：
1. focus on analyzing single tensor characteristics (lifetime, tensor size), struggle to adapt to the diverse requirement of different DNN models.
2. Global optimization solution is NP-hard
3. Need real-time monitoring (reply pn static analysis, 说autoTM和STR用brute-force mathematical techniques to caculate allocation strategies)
方法：offline+online training.
1. group micro tensors with similar access patterns into the same page.
2. allocate large tensors with comparable access frequencies to sequential pages.
3. Adapt tensor heat to the current state.
假设：CPU / GPU

其他知识：memory access patterns during DNN training can be profiled using computational graphs. (static/dynamic)

面向LLM：
[InfiniGen](All/InfiniGen.pdf) OSDI'24
[FlexGen](All/FlexGen.pdf) ICML'23

