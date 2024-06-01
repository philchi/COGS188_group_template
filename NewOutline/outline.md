## Synthetic Gradient Outline

### Ways to Speedup/Enable Large Scale Trainning
Admit that this field has already become a huge separated field from machine learning. As far as I know, we have four different kinds of ways:
- Data Parallelism
- Model Parallelism
- Tensor Parallelism
- Zero Redundancy Optimizer (ZeRO)
And some variants that aim to combine several different strategies.

We only wants to briefly descibe two of them:
- Data Parallelism
  - Each GPU contains the full model
  - Seperate each batch across GPUs (i.e. If we have bacth_size = 16 and 4 GPUs, each GPU will only process a batch_size = 4)
  - Gradients and parameters are synchronized across GPUs after all GPUs finish forward pass
  - vRAM consuming, usually will speed up the process a lot
- Model Parallelism
  - Slice model across several GPUs
  - Each GPU will process the input sequentially (Huggingface provides a great visualization)
    ![alt text](image.png)
    *From https://huggingface.co/docs/transformers/v4.17.0/en/parallelism*
    - Note: The bubble one sometimes will be referred as **pipeline solution** and we have even more advanced pipelines (interleaved pipeline) shown as below (Also from Huggingface)
    ![alt text](image-1.png)
  - Use less vRAM (i.e. If the model training process takes up to 16G vRAM, if you have 2 GPUs, each GPU only needs 8G to train the model)
  - Note:
    - Synthetic Gradient is also a kind of Model Parralelism
    - The pipeline used in synthetic gradient can almost reach 0 idle time for every single GPU.

### Problems with Backpropagation (Why Idle?)
- Three locks
  - Forward lock
  - Backward lock
  - Update lock
  - Notes:
    - Update lock and backward lock are mostly the same. The only difference is not all algorithm has a backward process. (i.e. Hopfield network) We can think in this way: backward lock is the process of computing gradients is locked, and update lock is the process of updating parameters is locked.
    - These three locks might not be a major problem for small neural netwoks but it is definitely a problem for huge neural networks (i.e. GPT-3 with 96 attention blocks)

### Synthetic Gradient Approach
- Math concepts
  - We should try to cover a bit simple math behind synthetic gradient...
  - Use a simple three-layered neural network to demonstrate the basci concepts of synthetic gradient
    - What synthetic gradient is trying to imitate
    - A interesting problem lies in synthetic gradient ---- the true gradient only backpropagates back to the layer before the last layer. The rest of the layer (including the synthetic gradient layers) are all updating based on fake gradients
      - Surprisingly, it still works well.
    - The training process of synthetic gradient (this can be shown by doing the math)
- How synthetic gradient accelerate training
  - Mainly by removing idle times
  - We could explain why it works in theory, but it is impossible for us to demonstrate how it works in practice ---- we only have one GPU and we don't have enough time to train a huge neural network
  - We could mention these things:
    - Synthetic gradient layer, though extremely simple (i.e. one or two layers perceptron, I don't even want to call it a neural network), will still introduce some extra cost
    - How long the idle time is? How much extra cost is needed?
    - If extra cost < idle time, then synthetic gradient can accelerate training process.
      - Usually a large neural network can satisfy the conditions
  - We could refer back to the visualization of pipeline shown above (those two visualizations from huggingface)
- Limitations
  - Synthetic gradient is not a magic, to accelerate training, one has to had more than one GPU
  - Definitely not an optimal way for small neural networks ---- especially the extra cost introduced are relatively high comparing to the network itself (i.e. the extra cost is nearly 10% of the cost of neural network itself)
  - Always note that even though synthetic gradient works, it is still using fake gradients. That is to say, it will result in a longer trainning time and usually a lower, though acceptable, accuracy.
  - Higher variance ---- not as stable as regular gradient descent.
    - Probably we should explain more on this when mentioning cDNI?

### Demo
As mentioned above, the demo is mainly aimed to show the synthetic gradient works.

### Model Acceleration and Beyond
- cDNI
  - The generation of synthetic gradient is conditioned by labels.
    - Avoid the math... P(X|Y=y) stuff...
  - Incredibly effective (speaking of the accuracy)
  - Should be able to implement by modifying part of the demo code (possibly as a take-home exercise?)
- Fully decoupled neural network (or, forward and backward decoupled)
  - Can be combined with cDNI, at least they are combined in the paper. (figure 7)
- Multi-model network
  - i.e. RNNs running on different time-clocks
- Synthetic gradient applied to CNNs directly (the demo approach requires flatten)
  - Use CNNs as synthetic gradient generator
    - Without max_pooling
    - With resolution preserved paddings