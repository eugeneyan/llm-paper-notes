# Mixture Of Experts Explained

> URL : https://huggingface.co/blog/moe

## Introduction

A Mixture of Experts (MOE) model involves two main things - training sub networks which eventually specialize certain tokens/tasks and a Router which learns which sub network to route a token to. For transformers, We implement this using a MOE layer which replaces the traditional FFN component of an attention block. 

Fundamentally, using a MOE model means we are trading VRAM for compute because it allows us to scale up the number of total parameters in our model while keeping the number of active parameters constant. Therefore, the compute to run inference ( in terms of FLOPs ) remains constant.

![Mixture Of Expert Layer](../assets/MOE-Block.png)

Here are some important characteristics of Mixture of Expert networks

- **Sparse**: Not all of the networks weights are connected to one another ( due to experts being seperate sub network that don't share parameters )
- **Training**: They can be trained faster because the computational graph is smaller due to lower number of nodes involved in each forward pass. [MOE-Mamba](https://arxiv.org/abs/2401.04081) reaches the same performance as Mamba in 2.2x less training steps while preserving the inference performance gains of Mamba against the Transformer.
- **High VRAM**: Since every expert must be loaded into memory for the model to run an inference step
- **Difficult to Fine-Tune**: These models seem to overfit quite easily so fine-tuning them for specific tasks has been difficult.
- **Challenging to Optimize**: Complex to perform - if we load balance requests to the wrong expert due to the appropriate expert being overwhelmed, then quality of response degrades. Possibility of wasted capacity if specific experts are also never activated.

We can also see that the performance of a MOE network increases as the number of **total** parameters increases. Note here that **total** parameters is not the same as **active** parameters. Total parameters refer to the number of parameters in the MOE model while active parameters only refer to the number of parameters involved in an inference step.

![MOE Parameter Scaling Relationship](../assets/MOE%20Scaling%20Law.png)



MOEs can also be implemented in other architectures, even RNNs and LSTMs.

![MOE RNN](../assets/MOE%20RNN.png)

## Architecture

The key things to note in a MOE model is going to be the routing mechanism. This controls the expert which we eventually dispatch the token to.

### Routing

> Most networks tend to set k = 2, which means that we combine the outputs of at most 2 experts. Some infrastructure providers such as Fireworks have chosen to provide k=3 mixtral in certain cases. 
>
> Fundamentally, this is a hyper-parameter that needs to be tuned. There are diminishing gains and latency trade-offs that will arise as we increase the number of experts. This is a topic of active study.

We utilise a learned gate network to determine the specific expert to send the token to
$$
y = \sum_{i=1}^nG(X)_i E_i(x)
$$


There are a few different ways to decide how to sample the experts to be chosen. They are

1. Top-k : Use a softmax function based on the output of the addition and normalization component of the attention block

2. Top-k with noise: Add some noise before applying softmax and sampling

3. Random Routing : Softmax for the first expert and then random sampling of the second based on softmax outputs

4. Expert Capacity : Calculate which experts are avaliable based on the average number of tokens to process per expert, then define a capacity multiple (Eg. each expert has capacity limit of 1.5x) - see below where C represents the capacity multiple, T the number of tokens, N the number of experts and LF the token capacity of an expert

   $$
   LF = \frac{C\times T}{N}
   $$

Note that we want to make sure each expert has a roughly equal distribution of tokens to proccess because of two main reasons

- Experts can be overwhelmed if they keep getting chosen to proccess tokens
- Experts will not learn if they never recieve tokens to proccess

Other methods that have been proposed includes 

- Expert Choice by Zhou et Al ( 2022 ): Allow experts to select the top t tokens from each sequence and then process it
- Soft mixtures of experts by Puigcerver et Al : In this model, experts act on *sequences* not tokens: each expert processes a weighted combination of all of the tokens in the input sequence.

#### Loss Functions

There are two main loss functions which we use when training a MOE network

1. Auxilliary Loss : Encourage each expert to have equal important and an equal number of training examples
2. Z-Loss : Penalize large logits entering the softmax function, therefore reducing potential routing errors

The reason why we want to penalise large logits is because of the issues with rounding errors in the routers. Switch Transformers experiment with casting different parameters to different datatypes in order to deal with this more efficiently.

## Training

### Fine-Tuning

Sparse models are going to benefit more from smaller batch sizes and higher learning rates but the models tend to overfit easily.

Models seem to memorise the training data - hence performing well on knowledge-heavy tasks such as TriviaQA while struggling with reasoning-heavy tasks such as SuperGLUE. 

![MOE vs Dense Model](../assets/MOE%20vs%20Dense.png)

However, when trained on instruction tuned data, we can see that MOEs see an greater increase in performance as compared to their dense counterparts. This can be seen below where we observe a greater increase in the eval score of the MOE model when it is finetuned on single-task instruction data.

![Instruction Tuning MOEs](../assets/MOE%20Instruction%20Tuning.png)

Data also seems to suggest that dropout probabilities within each expert has a moderate, more positive effect. Other interesting tricks include using upcycling - where we initialise an expert from the weights of the feed forward network.

This seems to speed up the training process by a significant proportion.

## Inference

Note that when we are refering to MOE inference, this refers to the inference per token.

MOE systems are challenging to run inference for because we cannot predict the load on each expert ahead of time. If there are multiple consecutive tokens that are related to each other in the sequence, this will result in an oversubscribed expert. Alternatively, if an expert is never chosen, then we have wasted compute that is just idly waiting for use.

### Running Things in Parallel

We have the four following ways to achieve parallelism.

- **Model parallelism:** the model is partitioned across cores, and the data is replicated across cores.
- **Data parallelism:** the same weights are replicated across all cores, and the data is partitioned across cores.
- **Model and data parallelism:** we can partition the model and the data across cores. Note that different cores process different batches of data.
- **Expert parallelism**: experts are placed on different workers. If combined with data parallelism, each core has a different expert and the data is partitioned across all cores

![MOE Parrallelism](../assets/MOE Parallel.png)

### Other Approaches

1. **Distillation**: Distil our MOE model into a dense equivalent. With this approach, we can keep ~30-40% of the sparsity gains.  Fedus et al (2021), for example, compare a sparse mixture of experts model to a dense T5-Base model that is 100 times smaller but is able to preserve the sparsity gains when distilled using a MOE T-5 model. 
2. **Modify Routing**: Route full sentences or tasks to an expert so that more information/context can be extracted
3. **Aggregation of MOE**: Merging the weights of the expert, reducing parameters at inference time.
4. **Custom Kernels**: Exploring new ways to batch the operations to take advantage of GPU parallelism. This is explored in Megablocks which expresses MoE layers as block-sparse operations that can accomodate imblaanced assignments in matrix mult ( when experts have diff utilisation )

## Challenges

### Expert Specialisation 

Expert Specialization seems to be on the token rather than sequence level

![Token Specific MOE](../assets/MOE Expert Routing.png)

We can see a similar example in the Mixtral MOE paper where they show the following diagrams

![Mixtral MOE Expert Routing](../assets/MOE%20Mixtral%20Tokens.png)

![Mixtral MOE Expert Routing](../assets/MOE%20Expert%20Choice.png)



## Examples

There are a variety of different MOE models that have recently been developed such as Mixtral 8x7b and Phixtral. 

### Mixtral 8x7b

Mixtral 8x7b uses a collection of Feed Forward Networks ( 8 Experts with 2 hidden layers ). It doesn't have 8x Mixtral 7Bs

```json
MixtralForCausalLM(
  (model): MixtralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MixtralDecoderLayer(
        (self_attn): MixtralAttention(
          (q_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear4bit(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear4bit(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MixtralRotaryEmbedding()
        )
        (block_sparse_moe): MixtralSparseMoeBlock(
          (gate): Linear4bit(in_features=4096, out_features=8, bias=False)
          (experts): ModuleList(
            (0-7): 8 x MixtralBLockSparseTop2MLP(
              (w1): Linear4bit(in_features=4096, out_features=14336, bias=False)
              (w2): Linear4bit(in_features=14336, out_features=4096, bias=False)
              (w3): Linear4bit(in_features=4096, out_features=14336, bias=False)
              (act_fn): SiLU()
            )
          )
        )
        (input_layernorm): MixtralRMSNorm()
        (post_attention_layernorm): MixtralRMSNorm()
      )
    )
    (norm): MixtralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```



### Mixtral Bits

Mixtral upsampled the proportion of multilingual dataset when pre-training. This in turn increase the ability of the model to perform well on multilingual benchmarks while mantaining a high accuracy in English.

# Relevant Resources/Useful Links

1. [MOEs by David Lakha](https://blog.javid.io/p/mixtures-of-experts) : Good walkthrough and links to more papers 
2. [MOE hardware requirements](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/3) : Forum page to understand hardware requirements to run an MOE system for inference
3. [FasterMOE](https://dl.acm.org/doi/10.1145/3503221.3508418) : Explores how to speed up MOE inference by examining a variety of different factors and blockers in normal MOE inference, resulting in a 17x speedup with their suggested changes.
4. [Upcycling MOEs](https://arxiv.org/abs/2212.05055): Explores how to speed up MOE training by initialising experts from original FFN network weights
5. [Instruction Tuned MOEs](https://arxiv.org/abs/2305.14705): Using 