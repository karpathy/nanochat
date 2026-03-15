---
title: "[Jan 7 2026] nanochat miniseries v1"
summary: ""
status: active
source: https://github.com/karpathy/nanochat/discussions/420
source_date: 2026-01-07
last_updated: 2026-01-07
---

# [Jan 7 2026] nanochat miniseries v1

> Source: [https://github.com/karpathy/nanochat/discussions/420](https://github.com/karpathy/nanochat/discussions/420)


**Why miniseries**. The correct way to think about LLMs is that you are not optimizing for a single specific model but for a family models controlled by a single dial (the compute you wish to spend) to achieve monotonically better results. This allows you to do careful science of scaling laws and ultimately this is what gives you the confidence that when you pay for "the big run", the extrapolation will work and your money will be well spent. For the first public release of nanochat my focus was on end-to-end pipeline that runs the whole LLM pipeline with all of its stages. Now, I'm coming back around to flesh out some of the parts that I sped through, starting of course with pretraining, which is both computationally heavy and critical as the foundation of intelligence and knowledge in these models.

**Miniseries v1**. In nanochat, that single dial is the depth of the model. For example, d12 is my favorite model (it's the size of GPT-1!) - it has 12 layers and currently trains in ~6 minutes. The setting of depth determines the number of channels in the Transformer (via the constant "aspect ratio"), and in turn the number of parameters and flops per token of the Transformer, and the optimization hyperparameters (the learning rate in particular) and finally via scaling laws analysis the horizon of training to obtain a "compute optimal" model (more on this below). As of the latest commit the script [miniseries.sh](https://github.com/karpathy/nanochat/blob/master/miniseries.sh), sweeps out the family of nanochat models from d10 to d20. All of these fit into a single 8XH100 node at the training batch size of `2**19 = 524,288` tokens without having to reach for micro batches and gradient accumulation. nanochat already supports gradient accumulation and I've trained much larger models (e.g. d34 recently), but I wanted to focus on this simplest setting first. The wandb plots look like this. x-axis is flops and y-axis is validation bpb (bits per byte, i.e. loss):

<img width="1120" height="393" alt="jan7_runs2" src="https://github.com/user-attachments/assets/65bd27e7-a8d5-4092-8215-9bf585da61af" />

What you're seeing here are models d10...d20. These 11 models took ~4 hours back to back on my trusty 8XH100 node to train for ~$100 of total cost. If your code, architecture and optimization is properly arranged and you did your scaling laws right these curves should not intersect. Each one represents the unique, compute optimal way to reach a target validation loss.

**Comparison to GPT-2/GPT-3 miniseries**. I did not want to use the validation loss to compare models because while it is simple, it can be subtle and deceiving. For example, [modded nanogpt](https://github.com/KellerJordan/modded-nanogpt) (which I otherwise love) merged a few changes that I thought were mildly gaming the metric (e.g. using very long sequences and batch size 1). It's a bit subtle but stretching out your validation batches into one long row (i.e. batch size B=1) just means there are fewer tokens with cropped contexts at the first few columns of your (B, T) batches when B>>1. This basically changes the validation loss by increasing the amount of context for many of these tokens, so it's not apples to apples, and the resulting "improvement" is not real. In addition, it's a bad form of comparison to GPT-2 and GPT-3 models because they were pretrained on a very different and unknown data mix distribution, so comparing FineWeb loss is not fair or informative. Only actual metrics are real and comparable. Earlier in the year I stumbled by the [DCLM paper](https://arxiv.org/abs/2406.11794) where they presented a nice ensemble metrics over a lot of different datasets. The metric is called CORE metric and it incorporates performance across 22 nice and high quality datasets. DCLM code had a complicated and bloated way of calculating it, so I stripped it all the way to a single, simple, dependency-free [file](https://github.com/karpathy/nanochat/blob/master/nanochat/core_eval.py) that evaluates the CORE metric given a model. Then we can chart a nice, valid comparison of our miniseries v1 models to GPT-2 and GPT-3 (more on how I calculated/estimated their CORE scores below) where the x axis is resource spend (FLOPs, time) and y axis is CORE score. To get $ as the x-axis, simply take the time (hours) and multiply by $24 (as the cost of 8XH100 is $3/GPU/hour X 8 GPUs = $24/hour).

<img width="1589" height="690" alt="jan7" src="https://github.com/user-attachments/assets/bf4325f7-b83d-4874-be56-ca4088d5b9bc" />

The goal for miniseries v2 is now simple: to further optimize the pretraining code and to lift up (and ideally tilt!) this line; to get more bang for the buck.

---

## Details

**Scaling laws**. One of the important and trickier aspects of getting this to work is doing a good job with your scaling laws (see [Kaplan et al.](https://arxiv.org/abs/2001.08361) and [Chinchilla/Hoffmann et al.](https://arxiv.org/abs/2203.15556)). The problem essentially is as follows. Suppose I want to train a d12 model. How many iterations should I train it for? Remember that at this small scale we are in the infinite data regime so there are no concerns of overfitting and therefore it makes no sense to, for example, train until your validation loss starts climbing. In the infinite data regime, the validation loss keeps going down indefinitely as you train longer (it just starts to level off slowly), moreover your train loss is basically equal to your validation loss - no overfitting. The answer to the problem is that the question is not quite right - you don't really want to train a d12. Instead, you have a certain compute budget of FLOPs (e.g. I want to run my cluster for exactly one day) and you want the lowest achievable loss. The real question then is: should you train a small model for many iterations or should you train a bigger model for fewer iterations? Scaling laws are a way of determining how to map from the single variable you have control over (the total number of flops) to the optimal setting of N (the number of parameters of your model) and D (the number of tokens you will train for, which is trivially related to the number of iterations or the length of time given a fixed batch size per step of the optimization).

Given a nanochat model of a certain depth, the way to calculate its flops is as follows:

```python
    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, the term 12 * l * h * q * t accounts for key @ query matmul flops inside attention.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
```

And the way to calculate its parameters is simply as:

```python
    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams
```

Next we use the `--target_flops` of `base_train.py` to fix the flops to a specific target (e.g. 3e18) and run models of a few depths. The code will automatically scale the number of iterations so that it exactly gets to your desired target flops. Small depths train long, large depths train short. However all of these models of different depths will end up costing exactly 3e18 FLOPs. The individual runs will look like this:

<img width="1098" height="371" alt="scaling_3e18" src="https://github.com/user-attachments/assets/8874d83e-c2cc-4be4-acd6-572b35dbf384" />

You see how the big models (e.g. d20, brown) ran for very iterations and small models (e.g. d8, magenta) ran for many iterations. All of these models cost the exact same amount of FLOPs, but clearly somewhere in the between one of them (d16 here) struck the correct balance and reached the lowest loss. That specific of model size and training length is compute optimal. When you repeat this process for a few FLOP budgets, you get surprisingly nice U shapes where for each one there is a concrete setting of the model size that is compute optimal:

<img width="1590" height="489" alt="scaling3" src="https://github.com/user-attachments/assets/f352f376-1519-452c-88e5-2cdc609cae7f" />

When you look at the optimal points (stars) after a quadratic fit you get:

```
FLOPs        Params          Tokens          Ratio      Val BPB   
-----------------------------------------------------------------
1e+18        136,706,248     1,116,751,173   8.2        0.9781    
3e+18        240,236,324     1,880,909,903   8.1        0.9202    
6e+18        330,186,832     2,711,097,322   8.5        0.8874   
```

For comparison, here is the same plot from the [Chinchilla paper](https://arxiv.org/abs/2203.15556) except with more compute:

<img width="1211" height="350" alt="chinchilla_scaling" src="https://github.com/user-attachments/assets/c1a7be3c-48b1-4c2a-9c95-4ea70c7d5819" />

Now there are a few important things to note. First, notice how (exactly like Chinchilla), the optimal number of parameters and tokens to train for is proportional to compute C to the power of ~0.5. First, sanity check they add up to ~1 because C ~= 6ND. But more importantly, they are equal to each other. This is a remarkable result that we reproduce, it means that parameters and tokens are on equal footing w.r.t. compute optimal models. For example if you double (2X) the compute budget, this is saying you should 1.41X your parameters and 1.41X your number of tokens (1.41 ~= sqrt(2)). In particular and most importantly, the fact that they are equal means that the optimal ratio of D:N is *constant*, regardless of the compute budget C. This is because if we model the number of parameters as N = k₁ · C^a and tokens D = k₂ · C^b, then D/N = (k₂/k₁) · C^(b-a) and so if a = b = 0.5, then C^(0.5 - 0.5) = C^0 = 1 = constant! So the optimal ratio of D/N = k₂/k₁ is constant regardless of the compute level of interest C. Note that this really didn't have to be the case, it could have been something else and complicated and a function of the model size (the original scaling laws paper of [Kaplan et al.](https://arxiv.org/abs/2001.08361) found this to be the case incorrectly due to a major bug in learning rate decay), but nature decided that compute optimal Transformers fall exactly on a straight line (what???). [Chinchilla](https://arxiv.org/abs/2203.15556) pointed it out and nanochat reproduces this surprising finding. In any case, practically speaking this is huge because it means that we have a single constant telling us the optimal ratio between D and N, and therefore we can simply have a `target_param_data_ratio` in nanochat base_train script, which calculates the optimal number of tokens to train for regardless of the depth of the model. In Chinchilla, they empirically measure k₂/k₁ to be 20. In nanochat, when you do the fits you actually get something much lower: 8. It's possible that some specifics of nanochat (the Muon optimizer, or...?) make it so that nanochat prefers bigger models trained shorter. Or it could be an artifact of the smaller model sizes we're looking at here. In any case, we now know how long we should train any given model. You take the number of its parameters N, you multiply by 8 to get the number of target tokens D, then divide D by the batch size (~0.5M) to get your number of iterations, done. We're now only training compute optimal models.

**Hyperparameter sweeps**. I ran a few more tuning sweeps that I won't spend a lot of time on. The learning rates are close to optimal after small nudge to the embedding learning rate. The warmdown ratio was the biggest surprise and I nudged it 0.2 -> 0.4. Sequence length of 2048 turns out to be quite good, balancing context length and document diversity in our batch of 0.5M tokens. The batch size 0.5M is a little bit on the larger size and purely flops-wise should be a little bit smaller (~half), but wallclock-wise is good as is. All this to say that I did some basic tuning for miniseries v1, but by no means exhaustive and there are still many ideas to try.

**GPT-2 / GPT-3 CORE scores**. Another challenge was calculating CORE scores for the GPT-2 and GPT-3 miniseries. GPT-2 miniseries was easy because the models are available and were released. So you can just download the models and run the eval. GPT-3 miniseries I had to get more creative because the models were never released. But we do have the paper with their evaluation results in the tables. I posted the full approach to a [jupyter notebook](https://github.com/karpathy/nanochat/blob/master/dev/estimate_gpt3_core.ipynb) but basically I found 6 tasks that are both in the CORE metric *and* reported in the GPT-3 paper in a very similar evaluation setting. They are `['HellaSwag 0-shot', 'LAMBADA', 'HellaSwag 10-shot', 'PIQA', 'ARC Easy', 'ARC Challenge']`. I then use the GPT-2 models for calibration, meaning that I trained a simple model that takes the performance on these 6 tasks and estimates the CORE score (of 22 tasks) using 3 different approaches. The fact that these 6 are solid evals and that the points lined up very nicely gave me confidence that these CORE scores are not far off.

After all this work we get our targets:

  | GPT-2  | Params | Calculated CORE  | GPT-3  | Params | Estimated CORE  |
  |--------|--------|-------|--------|--------|-------|
  | Small  | 124M   | 0.114 | Small  | 125M   | 0.148 |
  | Medium | 355M   | 0.185 | Medium | 350M   | 0.216 |
  | Large  | 774M   | 0.215 | Large  | 760M   | 0.266 |
  | XL     | 1.6B   | 0.257 | XL     | 1.3B   | 0.291 |
  | -      | -      | -     | 2.7B   | 2.7B   | 0.329 |
  | -      | -      | -     | 6.7B   | 6.7B   | 0.361 |
  | -      | -      | -     | 13B    | 13B    | 0.385 |
  | -      | -      | -     | 175B   | 175B   | 0.427 |

(Note 1: You'll notice that GPT-3 at the same parameter counts as GPT-2 is a slightly better model with stronger performance due to various improvements to data, architecture, optimization and they are also trained for a lot more tokens (300B) compared to GPT-2's estimated token budget of somewhere around ~100B tokens. Note 2: All of these are CORE scores v1, not v2 ([iykyk](https://github.com/mlfoundations/dclm) otherwise nvm)).

**Miniseries v1 CORE scores** Second, here are the nanochat miniseries v1 in their full detail:

|   depth |   model_dim |   params_M |   tokens_B |   val_bpb |   CORE |   train_time_min |
|--------:|------------:|-----------:|-----------:|----------:|-------:|-----------------:|
|      10 |         640 |         91 |       0.73 |    1.0312 | 0.071  |              5   |
|      11 |         704 |        112 |       0.89 |    1.0096 | 0.0918 |              6.6 |
|      12 |         768 |        135 |       1.08 |    0.9825 | 0.1059 |              7.8 |
|      13 |         832 |        163 |       1.3  |    0.9644 | 0.1015 |             11.2 |
|      14 |         896 |        194 |       1.55 |    0.9437 | 0.1185 |             13.2 |
|      15 |         960 |        229 |       1.83 |    0.9257 | 0.1158 |             17.4 |
|      16 |        1024 |        268 |       2.15 |    0.9101 | 0.1332 |             21.4 |
|      17 |        1088 |        313 |       2.5  |    0.8952 | 0.1518 |             33.9 |
|      18 |        1152 |        362 |       2.9  |    0.8817 | 0.1611 |             37.5 |
|      19 |        1216 |        417 |       3.33 |    0.8687 | 0.1659 |             53   |
|      20 |        1280 |        477 |       3.82 |    0.8572 | 0.1708 |             59.5 |

**NOTE**: Do not be confused w.r.t. the v1 miniseries and previous nanochat models I have trained so far in the rest of the discussions. Those models were trained with D:N ratio of 20 (Chinchilla), these models are using 8. So they are less trained (at each depth), but compute optimal for their respective validation loss.

We can also take the models d12+ (discarding some of the smaller models due to fear of outliers at that tiny of a scale) and do a fit predicting (total) parameters -> CORE of the asymptotic form to get a fit `CORE = 1.0000 - 3.7555 * FLOPs^(-0.0344)`. With this, we can extrapolate to see what we need to reach all of the GPT-2 and GPT-3 models:

Depth to match GPT-2/3 CORE (d>=12 fit, 12:1 D:N ratio, 8xH100 @ $3/GPU/hr):

| Model               |   CORE | Depth   | Params   | Tokens   |   FLOPs | Time (8xH100)   | Cost     |
|:--------------------|-------:|:--------|:---------|:---------|--------:|:----------------|:---------|
| GPT-2 (124M)        |  0.114 | d14     | 179M     | 1.4B     | 1.7e+18 | 8 min           | $3       |
| GPT-2 Medium (355M) |  0.185 | d22     | 599M     | 4.8B     | 2e+19   | 1.6 hrs         | $38      |
| GPT-2 Large (774M)  |  0.215 | d28     | 1.0B     | 8.1B     | 5.8e+19 | 4.6 hrs         | $111     |
| GPT-2 XL (1.6B)     |  0.257 | d38     | 2.2B     | 17.8B    | 2.9e+20 | 22.7 hrs        | $546     |
| GPT-3 Small (125M)  |  0.148 | d17     | 319M     | 2.6B     | 5.5e+18 | 26 min          | $10      |
| GPT-3 Medium (350M) |  0.216 | d28     | 1.0B     | 8.3B     | 6.1e+19 | 4.8 hrs         | $116     |
| GPT-3 Large (760M)  |  0.266 | d41     | 2.7B     | 21.3B    | 4.2e+20 | 1.4 days        | $790     |
| GPT-3 XL (1.3B)     |  0.291 | d51     | 4.4B     | 35.3B    | 1.2e+21 | 3.8 days        | $2.2k    |
| GPT-3 2.7B          |  0.329 | d71     | 9.6B     | 77.2B    | 5.7e+21 | 18.9 days       | $10.9k   |
| GPT-3 6.7B          |  0.361 | d95     | 19.3B    | 154.6B   | 2.4e+22 | 78.0 days       | $44.9k   |
| GPT-3 13B           |  0.385 | d119    | 33.4B    | 267.6B   | 7.3e+22 | 238.6 days      | $137.5k  |
| GPT-3 175B          |  0.427 | d181    | 91.8B    | 734.1B   | 5.7e+23 | 1869.0 days     | $1076.5k |

Now, I wouldn't really read into this too much because we're doing *a lot* of extrapolation over many orders of magnitude of flops based on very few datapoints. And we're optimistically assuming the same utilization at scale as that of the d20 run that fully utilizes the 8XH100 box to estimate the time and cost. But still, it's encouraging to see that e.g. as a sanity check, the predicted FLOPs needed to get to GPT-3 level is 5.7e23 (the real amount needed for the GPT-3 run was ~3e23). If these numbers are to be trusted, then we're in a pretty decent spot, but still with quite a bit of room for improvement.

For any Q&A please feel free to use the discussion below, alternatively find me on the Discord channel [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031).