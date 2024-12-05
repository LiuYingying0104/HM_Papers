Instruction：
读introduction和abstract：
1. 在这个领域内最常被引述的方法有哪些？ 
   1. vDNN
   2. AutoTM
   3. Sentinel
   4. SuperNeurons
   5. cuDNN
   6. ……
2. 这些方法可以分成哪些主要派别？
   1. re-compute (rematerialization)
   2. swapping (tiering)
   3. compression (model compression/data compression)
   4. memory optimization (change operator/ reuse memory)
   5. distributed training
   6. memory allocation
3. 每个派别的主要特色（含优点和缺点）是什么？
   感觉1和2的优势主要在于计算需要花的时间和swapping需要花的时间的trade-offline；
   1和2是trade memory for data movement or computation time.
   1. 1的优势在于不需要PCIe bandwidth，有人提到觉得recompute在复杂依赖的layer上表现得不好, 对于dynamic memory allocation，没法直接用 有bound，recompute会攻击paging说：paging is energy-intensive and often less efficient than rematerialization
   2. 不过swapping可以做到不依赖于profiling或者static analysis，不过要考虑PCIe bandwidth，我觉得这个东西是需要online去learn来的
   swap的另一个优点：CPU memory比GPU memory更大更便宜/ 2.GPU 可以边通信边计算/ 3. GPU 和 CPU communication bandwidth 可以满足
   如果是纯dynamic的话，跟data tiering的OS-based的方法比较像，缺点就是 don't take into account future information about the data use and semantic information from the Application
   3. compression的话如果是data compression且是无损的，挺好的，但应该也是有限的 有bound
   如果是model compression会影响精度，我们目标是不影响精度, 并且weights在整个当中占比不高， 需要heavy hyper-parameter tuning
   4. change operator 和 reuse memory应该都是需要对program进行比较精细的分析，（不知道这个运行时间的量级是什么样的）有的是用user-level APIs，但是调用API programming的时候会比较麻烦。也有人会把这一类叫做computational graph analysis；memory reuse也有叫memory sharing 有bound
   还有一类是data tiering，这种应该普适性更强一些，不局限于DNN training。
   5. distributed 这个类别的话有些依然需要把model在GPU上replicate，对于一些model本身就塞不进去的还是不行，而且会有communication overhead, 我个人觉得programming上也不是很简洁, bring communication overhead 
   enforce a non-trival bound on minimum number of GPUs for large models / 有bound, 总量还是有要求 / throughput--near linear/ cost--linear -> result: sub-optimal cost efficienct.
4. 这个领域内大家认为重要的关键问题有哪些？
   1. 能容纳的memory的上限（包括batch size也是一个指标）
   2. performance overhead (执行/training的总时间): cost(memory budget + time limitation), energy-efficient 
   3. 是否对本身task产生影响（accuracy）
   4. 会不会对其他正在进行的task有影响？***这一点可以看一下
   5. 是否是对单一的neural network进行优化，还是说可以普适于所有的network，甚至是所有的task
   6. 是否make assumption about model's structure
   7. 使用时single GPU / multiple GPUs / GPU clusters
   8. fragmentation的问题
   9. 是否依赖于computation graph（通常是static analysis），还是说依赖于dynamic profiling或者online的approach，是否需要硬件上的修改?
   10. 跟已经有的framework：pytorch或者tensorflow是否可以seamlessly衔接
   11. 如果要找optimization的话，那么运行它需要多久呢？
   12. 是否需要real-time monitoring？需要的话overhead是多少？
5. 有哪些特性是大家重视的优点？
6. 有哪些特性是大家在意的缺点？
7. 这些优点和缺点通常在哪些应用场合时会比较被重视？
8. 在哪些应用场合会比较不会被重视？

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