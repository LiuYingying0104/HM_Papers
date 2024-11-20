
[vDNN](vDNN.pdf) MICRO'16
要解决的问题：模型太大 GPU内存不够 reduce memory usage of neural network  
提到的其他方法1：network pruning/quantization  
觉得他们不好的点：weights占比在整个memory usage中很小/loss of accuracy  
提到的其他方法2：page-migration based virtualization [51][52][34]  
觉得他们不好的点：underutilize PCIe bandwidth/performance overhead  
方法：要么直接release，要么offload to CPU Memory 之后再prefetch回来  

[AutoTM](AutoTM.pdf) ASPLOS'20
要解决的问题：模型太大，DRAM不够
提到的其他方法1：用SSD做backing storage [13][14]
提到的其他方法2：memory management between CPU and GPU [8][44]
方法：提出一个framework，automatically move data between heterogenous memory devices. (compiler-based)
假设条件：computational graph is static (no data-dependent control behavior)
别的文章觉得它不好的点：kernel computation and its tensor accesses are coupled as one execution unit. 
Tensor-swap operation cannot be decoupled from its kernel, thereby losing an optimization opportunity of overlaping tensor swapping with an independent kernel's execution.

[SwapAdvisor](SwapAdvisor.pdf) ASPLOS'20
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

[Capuchin](Capuchin.pdf) ASPLOS'20
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
2. major memory footprint in DNN training 是来自于intermediate layer outputs [13][24]
3. 不基于computation graph （适用于imperative programming environment）
其他：文章把model的memory consumption分成三个类别：feature maps / gradient maps/ convolution workspace （model parameters占比很小）
Programming中有两个mode：1. eager mode（对应于imperative programming），不生成计算图，但是因为没有optimization/需要interpret python code，所以overhead大 （例子：PyTorch/Tensorflow2.0）
2. graph mode（对应于declarative programming）before execution，computation graph is built （例子：TensorFlow1.0）

[Sentinel](Sentinel.pdf) HPCA'21
要解决的问题：现在Heterogeneous Memory 越来越流行，用HM实现DNN training achieve larger memory capacity
提到的其他方法1：用 CPU-side system memory来扩容[5-11] / 用persistent memory来扩CPU的容 
based on DNN topology[7-10] / detailed domain knowledge [5][6][11]
觉得不好的地方：semantic gap between OS and memory management in deep learning framework (比如 lifetime不同可以放的位置不同/OS是page-level，而predict的是tensor-level，会造成fragmentation)[12][14]
不考虑lifetime的话，会把一些short-live的给挪动了[15]/hardware[16-18]
方法：dynamic profiling / OS / runtime-level profiling / performance model (让migration跟DNN training同时进行) 
其他：面向的是tensorflow

[ZeRO-Offload](ZeRO-Offload.pdf) ATC'21
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

[Zico](Zico.pdf) ATC'21
要解决的问题：reduce system-wide memory consumption for concurrent training （GPU Sharing）
提到的方法派别1：temporally multiplexing [26,43,44]
觉得这个派别不好的地方：大多数时候计算资源没有被充分利用
提到的方法派别2：spatially sharing [10]
觉得这个派别不好的地方：working set太大无法fit in GPU memory --> 所以对于这个派别来说，需要reduce memory footprint
方法： forward时候memory usage在增加，backward时候memory usage在减少，因此可以共享一个memory pool，应该是用一个GPU kernel去monitor memory allocate和release的event，然后再用scheduler去schedule
其他：面向的是tensorflow

[ZeRO-Infinnity](ZeRO-Infinity.pdf) SC'21 （这个是ZeRO-Offload的延续）
要解决的问题：模型太大 GPU内存不够
1. How do we support the next 1000x growth in model size, going from models like GPT-3 with 175 billion parameters to models with hundreds of trillions of parameters?
2. How can we make large models of today accessible to more data scientists who don’t have access to hundreds to GPUs？
3. Can we make large model training easier by eliminating the need for model refactoring and multiple forms of parallelism?
方法：在heterogeneous memory上（GPU,CPU,DRAM）上allow model scale without requiring model code refactoring / memory-centric tiling 解决单层过大的问题/ bandwidth-centric partitioning 根据device带宽来决定分区/ overlap-centric design (把offload跟communication尽量并行运算)

[TSPLIT](TSPLIT.pdf) ICDE'22
要解决的问题：DNN太大， device memory不够
提到的方法1：distributed system [20-22]
觉得不好的点：bring high communication costs/ increase system complexity
提到的方法2：model compression
觉得不好的点：affects final accuracy of the model/ require heavy hyper-parameter tuning
提到的方法3：recompute or swap [27][19][17] tensor-wise [33-36] [37-38]
觉得不好的点：在某些operations下，依旧没法trainable / 大tensor的交换很time-consuming
方法：break operation boundary （micro-tensor）/ computation graph profile
假设：应该也是基于静态的computation graph的

[TMOF](TMOF.pdf) HPCA'23
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

[DeepUM](DeepUM.pdf) ASPLOS'23
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

[G10](G10.pdf) MICRO'23
要解决的问题：hard to meet the scalability requirement of deep learning workloads
提到的其他方法1：expand limited GPU memory with flash memory. [67,68,61,19,21] 这些感觉是偏向technique方面的，bring Flash closer to GPUs
觉得他们不好的点：limited bandwidth by the PCIe interface
提到的其他方法2：提出特殊的针对DNN的方法 move data across heterogeneous memories [12,25,27,34]
觉得他们不好的点: complicates GPU memory management, hurt development productivity
方法：integrate host memory, GPU memory and flash memory into a unified memory space / 用compiler technique to characterize the tensor behaviors
看了代码，输入是转化过的code，类似于DNN网络结构信息，然后用静态分析等方法来计算
其他：因为需要融合了SSD，所以需要修改driver。但是NVIDIA driver不开源，因此需要用到simulator 
假设：single GPU，测试的是传统DNN和transformer

[HOME](HOME.pdf) IEEE Transaction on Computers 23
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

[Occamy](Occamy.pdf) DAC'23
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

[CachedArrays](CachedArrrays.pdf) IPDPS'24


[MAGIS](MAGIS.pdf) ASPLOS'24
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


面向LLM：
[InfiniGen](InfiniGen.pdf) OSDI'24
[FlexGen](FlexGen.pdf) ICML'23



Instruction：
读introduction和abstract：
1. 在这个领域内最常被引述的方法有哪些？
2. 这些方法可以分成哪些主要派别？
3. 每个派别的主要特色（含优点和缺点）是什么？
4. 这个领域内大家认为重要的关键问题有哪些？有哪些特性是大家重视的优点？
有哪些特性是大家在意的缺点？这些优点和缺点通常在哪些应用场合时会比较被重视？在哪些应用场合会比较不会被重视？

按派别进行阅读main body：按时间先后顺序阅读
1. 这篇论文主要假设是什么（跟现实情况是什么做对比）
2. 这些假设下论文的好处在哪里
3. 好处主要体现在哪些项目的简化上
这一个派别的主要发展过程、主要假设、主要理论依据、主要成果
这一派最适合什么时候使用、最不适合什么场合使用


不同的场景：
1. high-end data centers
2. common computing platform for researchers and developers

感觉很多都没有考虑OS方面的协作
Design Method:
1. Expert knowledge
2. Heurestic (Greedy, ILP(high algorithm complexity), PSO)
3. Meta-Heurestic (genetic)