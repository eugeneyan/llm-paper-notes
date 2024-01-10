# Mixture Of Experts Explained

> URL : https://huggingface.co/blog/moe

Some questions

1. Why use all FFNs, why not 2 FFNs, one LSTM or a RWKV block? Is there value in having experts all have the same architecture?
2. Difference between dropout and MOE -> Instead of top-k =2 why not just choose a random k from 1 -> 5 each time? This way we build more resilience in the network
3. Why is it only in the FFN? Why can't we throw out 4 diff attention blocks -> Mixture of Attention?
4. What is a dense model equivalent of a sparse moe model? Is it refering to the active parameter count or the total parameter count?

**tl;dr** : MOE is a architectural choice to route observations to subnetworks within a block. This allows us to scale up parameter counts by introducting more experts and hence capabilities of our network. However, this introduces new  challenges due to the higher parameter count to run inference with, training instabilities and inference-time provisioning of experts across devices.

## Introduction

A Mixture of Experts model involves two main things - training sub networks which eventually specialize certain tokens/tasks and a classifier which learns which sub network to route a token to. This is done by introducing Mixture Of Experts (MOE) layers. We want to train an MOE normally when we want to scale up parameters ( and capabilities ) without increasing the amount of compute required to run the inference step ( in terms of FLOPs ).

For a transformer, this means replacing every FFN layer with a MOE layer as seen before

![Mixture Of Expert Layer](../assets/MOE-Block.png)

Here are some important characteristics of Mixture of Expert networks

- **Sparse**: Not all of the networks weights are connected to one another ( due to experts being seperate sub network that don't share parameters )
- **Training**: They can be trained faster because the computational graph is smaller due to lower number of nodes involved in each forward pass. MOE-Mamba reaches the same performance as Mamba in 2.2x less training steps while preserving the inference performance gains of Mamba against the Transformer.
- **High VRAM**: Since every expert must be loaded into memory for the model to run an inference step
- **Difficult to Fine-Tune**: These models seem to overfit quite easily so fine-tuning them for specific tasks has been difficult.
- **Challenging to Optimize**: Complex to perform - if we load balance requests to the wrong expert due to the appropriate expert being overwhelmed, then quality of response degrades. Possibility of wasted capacity if specific experts are also never activated.

We can see that the sparse models also show consistent performance increases when parameter counts are scaled in the Switch Transformer Paper

![MOE Scaling](../assets/MOE%20Eval.png)

## Architecture

The key things to note in a MOE model is going to be the routing mechanism. This controls the expert which we eventually dispatch the token to.

### Routing

We utilise a learned gate network to determine the specific expert to send the input to
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

> Question: What does it mean when a model does worse in smaller tasks but did well in larger tasks? Also, what is the MOE graph supposed to represent?

![CleanShot 2024-01-10 at 20.26.10](/Users/admin/Library/Application Support/CleanShot/media/media_Sg8fySzcDi/CleanShot 2024-01-10 at 20.26.10.png)

However, there seem to be good results with recent attempts at instruction tuining so that might change things. 

### Other Methods

Data also seems to suggest that dropout probabilities within each expert has a moderate, more positive effect. Other interesting tricks include using up cycling - where we initialise an expert from the weights of the feed forward network.

This seems to speed up the training process by a significant proportion.

## Inference

it is challenging to run inference for MOE systems because we cannot predict the load on each expert ahead of time. This means that it is a real possibility that we will be unable to process all tokens in the sequence if our expert is unable to cope with the demand. We can however try to optimise the inference process by using some degree of parallelism.

In practice, a small number of experts are always allocated a large share of tokens; others are completely inactive. Machine translation is particularly bad, because there is high temporal correlation: if one sequence makes use of a particular expert, the probability that the next sequence will use that expert is higher.

![MOE Expert Distribution](../assets/MOE%20Expert%20Distribution.png)

There are specific experts here that are used significantly more than the others. Note that we have ~120+ experts so we expect utilisation to be < 0.01 if everything is fairly allocated.

### Running Things in Parallel

We have the four following ways to achieve parallelism.

- **Model parallelism:** the model is partitioned across cores, and the data is replicated across cores.
- **Data parallelism:** the same weights are replicated across all cores, and the data is partitioned across cores.
- **Model and data parallelism:** we can partition the model and the data across cores. Note that different cores process different batches of data.
- **Expert parallelism**: experts are placed on different workers. If combined with data parallelism, each core has a different expert and the data is partitioned across all cores

![MOE Parrallelism](../assets/MOE Parallel.png)

### Other Approaches

1. **Distillation**: Distil our MOE model into a dense equivalent. With this approach, we can keep ~30-40% of the sparsity gains.  Fedus et al (2021), for example, compare a sparse mixture of experts model to a dense T5-Base model that is 100 times smaller but is able to preserve the sparsity gains when distilled using a MOE T-5 model
2. **Modify Routing**: Route full sentences or tasks to an expert so that more information/context can be extracted
3. **Aggregation of MOE**: Merging the weights of the expert, reducing parameters at inference time.
4. **Custom Kernels**: Exploring new ways to batch the operations to take advantage of GPU parallelism

FasterMoE (March 2022) analyzes the performance of MoEs in highly efficient distributed systems and analyzes the theoretical limit of different parallelism strategies, as well as techniques to skew expert popularity, fine-grained schedules of communication that reduce latency, and an adjusted topology-aware gate that picks experts based on the lowest latency, leading to a 17x speedup.

Megablocks (Nov 2022) explores efficient sparse pretraining by providing new GPU kernels that can handle the dynamism present in MoEs. Their proposal never drops tokens and maps efficiently to modern hardware, leading to significant speedups. What’s the trick? Traditional MoEs use batched matrix multiplication, which assumes all experts have the same shape and the same number of tokens. In contrast, Megablocks expresses MoE layers as block-sparse operations that can accommodate imbalanced assignment.

## Challenges

### Expert Specialisation 

Expert Specialization seems to be on the token rather than sequence level

![Token Specific MOE](../assets/MOE Expert Routing.png)

We can see a similar example in the Mixtral MOE paper where they show the following diagrams

![Mixtral MOE Expert Routing](../assets/MOE%20Mixtral%20Tokens.png)

![Mixtral MOE Expert Routing](../assets/MOE%20Expert%20Choice.png)

> The expected proportion of repetitions in the case of random assignments is 1/8 = 12.5% for “First choice. Repetitions at the first layer are close to random, but are significantly higher at layers 15 and 31. The high number of repetitions shows that expert choice exhibits high temporal locality at these layers.
>
> Mistral Paper

## Examples

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

# Relevant Resources

1. [MOEs by David Lakha](https://blog.javid.io/p/mixtures-of-experts)