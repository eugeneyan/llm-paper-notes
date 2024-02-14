# llm-paper-notes
Notes from the [Latent Space paper club](https://www.latent.space/about#§components). Follow along or start your own!

---

1. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762):** Query, Key, and Value are all you need\* (\*Also position embeddings, multiple heads, feed-forward layers, skip-connections, etc.)

2. **[GPT: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf):** Decoder is all you need\* (\*Also, pre-training + finetuning)

3. **[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805):** Encoder is all you need\*. Left-to-right language modeling is NOT all you need. (\*Also, pre-training + finetuning)

4. **[T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683):** Encoder-only or decoder-only is NOT all you need, though text-to-text is all you need\* (\*Also, pre-training + finetuning)

5. **[GPT2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):** Unsupervised pre-training is all you need?!

6. **[GPT3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165):** Unsupervised pre-training + a few* examples is all you need. (\*From 5 examples, in Conversational QA, to 50 examples  in Winogrande, PhysicalQA, and TriviaQA)

7. **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361):** Larger models trained on lesser data\* are what you you need. (\*10x more compute should be spent on 5.5x larger model and 1.8x more tokens)

8. **[Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556):** Smaller models trained on more data\* are what you need. (\*10x more compute should be spent on 3.2x larger model and 3.2x more tokens)

9. **[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971):** Smoler models trained longer—on public data—is all you need

10. **[InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155):** 40 labelers are all you need\* (\*Plus supervised fine-tuning, reward modeling, and PPO)

11. **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685):** One rank is all you need

12. **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314):** 4-bit is all you need\* (\*Plus double quantization and paged optimizers)

13. **[DPR: Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906):** Dense embeddings are all you need\* (\*Also, high precision retrieval)

14. **[RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401):** Semi-parametric models\* are all you need (\*Dense vector retrieval as non-parametric component; pre-trained LLM as parametric component)

15. **[RETRO: Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426):** Retrieving based on input chunks and chunked cross attention are all you need

16. **[Internet-augmented language models through few-shot prompting for open-domain question answering](https://arxiv.org/abs/2203.05115):** Google Search as retrieval is all you need

17. **[HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496):** LLM-generated, hypothetical documents are all you need

18. **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135):** For-loops in SRAM are all you need

19. **[ALiBi; Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409):** Constant bias on the query-key dot-product is all you need\* (\*Also hyperparameter m and cached Q, K, V representations)

20. **[Codex: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374):** Finetuning on code is all you need

21. **[Layer Normalization](https://arxiv.org/abs/1607.06450):** Consistent mean and variance at each layer is all you need

22. **[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745):** Pre-layer norm, instead of post-layer norm, is all you need

23. **[PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347):** Clipping your surrogate function is all you need

24. **[WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568):** Asking the model to make the question harder is all you need\* (\*Where do they get the responses to these harder questions though?!)

25. **[Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)**: Iterative finetuning, PPO, rejection sampling, and ghost attention is all you need\* (\*Also, 27,540 SFT annotations and more than 1 million binary comparison preference data)

26. **[RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048):** Linear attention during inference, via RNNs, is what you need

27. **[RLAIF; Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073):** A natural language constitution\* and model feedback on harmlessness is all you need (\*16 different variants of harmlessness principles)

28. **[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538):** Noise in your softmax and expert regularization are all you need

29. **[CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020):** \*A projection layer between text and image embeddings is all you need (\*Also, 400 million image-text pairs)

30. **[ViT; An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929):** Flattened 2D patches are all you need

31. **[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442):** Reflection, memory, and retrieval are all you need

32. **[Out-of-Domain Finetuning to Bootstrap Hallucination Detection](https://eugeneyan.com/writing/finetuning/):** Open-source, permissive-use data is what you need

33. **[DPO; Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290):** A separate reward model is NOT what you need

34. **[Consistency Models](https://arxiv.org/abs/2303.01469):** Mapping to how diffusion adds gaussian noise to images is all you need

35. **[LCM; Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378):** Consistency modeling in latent space is all you need\* (\*Also, a diffusion model to distill from)

36. **[LCM-LoRA: A Universal Stable-Diffusion Acceleration Module](https://arxiv.org/abs/2311.05556):** Combining LoRAs is all you need

37. **[Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models](https://arxiv.org/abs/2311.09210):** Asking the LLM to reflect on retrieved documents is all you need

38. **[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682):** The Bitter Lesson is all you need

39. **[Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions](https://arxiv.org/abs/2309.10150):** The Bellman equation and replay buffers are all you need

40. **[Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674):** Classification guidelines and the multiple-choice response are all you need

41. **[REST^EM; Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)**: Synthetic data and a reward function are all you need

42. **[Mixture Of Experts Explained](https://huggingface.co/blog/moe#capacity-factor-and-communication-costs)**: MOE is a architectural choice to route observations to subnetworks within a block. This allows us to scale up parameter counts by introducting more experts and hence capabilities of our network. However, this introduces new  challenges due to the higher parameter count to run inference with, training instabilities and inference-time provisioning of experts across devices.

43. **[Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)**: ([GitHub Repo](https://discord.com/channels/822583790773862470/1197350122112168006), [LS Discord chat](https://discord.com/channels/822583790773862470/1197350122112168006/1199809444897361970))

44. **[Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)**: ([Presentation](https://docs.google.com/presentation/d/1EfNdcpRWqns8DvNIL-1bZdOI5tkbmcoBvMQxoMSbbI4/edit?usp=sharing)) A series of open source LLMs with completely reproducible datasets and checkpoints for LLM research. Novel studies (incl negative results) in memorization, data deduplication and data order, and gender debiasing.

45. **[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)**: ([GitHub Repo](https://github.com/lucidrains/self-rewarding-lm-pytorch), [LS Discord Chat](https://discord.com/channels/822583790773862470/1197350122112168006/1204880366993674251)): Instead of training reward models from human preferences, LLMs can be used to provide their own rewards during training (aka WITHOUT distillation from GPT4). Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613.

46. **[Leveraging Large Language Models for NLG Evaluation: A Survey](https://arxiv.org/pdf/2402.00838.pdf)**: [Slides](https://docs.google.com/presentation/d/17mbqFh8KFgn4SqRiKZmtjDn83N6d_jhqep8oPGbtB50/edit#slide=id.g2b89c4d23f6_0_329), [LS Discord Chat](https://discord.com/channels/822583790773862470/1197350122112168006/1207416409383108659) a survey paper of model and task eval techniques. Includes using Auto-J correlation instead of AlpacaEval which we liked.
