---
title: "Introducing nanochat: The best ChatGPT that $100 can buy."
summary: ""
status: active
source: https://github.com/karpathy/nanochat/discussions/1
source_date: 2025-10-13
last_updated: 2025-10-13
---

# Introducing nanochat: The best ChatGPT that $100 can buy.

> Source: [https://github.com/karpathy/nanochat/discussions/1](https://github.com/karpathy/nanochat/discussions/1)

> **Note:** This is the original Oct 2025 post. Several commands and script names are outdated. The current reference pipeline is [`runs/speedrun.sh`](../../runs/speedrun.sh). Differences are noted inline.

Ok so we just booted up an 8xH100 box from e.g. [Lambda GPU Cloud](https://lambda.ai/service/gpu-cloud). This is costing us about ~$24/hr, so there is no time to lose. 

## Environment setup

Clone the project:

```bash
git clone git@github.com:karpathy/nanochat.git
cd nanochat
```

We wish to train the best ChatGPT that $100 can buy, which we call a "speedrun". Reference the script [speedrun.sh](speedrun.sh), which is designed to just run right away on a blank box start to end. However, in this post I will step through it part by part so that I can comment in detail on all sections of it. We first have to make sure the new&hot [uv](https://docs.astral.sh/uv/) project manager is installed. Install uv, create a new virtual environment in `.venv`, get all the dependencies, and activate the environment so that when we type `python` we're using the virtual env python, not the system python:

```bash
# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
```

Next, we need to install Rust/Cargo so that we can compile our custom Rust tokenizer. I know - it's a bit much to have a new/custom tokenizer, unfortunately the Python version in my earlier [minbpe](https://github.com/karpathy/minbpe) project is way too slow and the huggingface [tokenizers](https://github.com/huggingface/tokenizers) is too bloated and confusing. So we have our own new tokenizer for training (tested to be equal to Python), but we will still use OpenAI's [tiktoken](https://github.com/openai/tiktoken/tree/main) for efficient inference. So here we go, compile our tokenizer:

```bash
# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Train the tokenizer

Next we need the pretraining data so that we can 1) train the tokenizer and 2) pretrain the model. The pretraining data is just the text of a lot of webpages, and for this part we will use the [FineWeb-EDU](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) dataset. Normally, we'd be able to just use huggingface `datasets.load_dataset()`, but I didn't like how it's too heavy, bloated and obscures some very simple logic, so I re-packaged the entire dataset into simple, fully shuffled shards that we can easily and efficiently access at will and re-uploaded the sample-100B version of it as [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle). On this page you can also preview example text in the dataset. Each shard is a simple parquet file of about 0.25M characters and takes up about 100MB on disk (gzip compressed). There are 1822 shards in total but we only need 240 of them to train a depth=20 model (more on this later). So let's download all of the data now. This is about ~24GB download here, but it's fairly zippy on a cloud box normally:

```bash
nanochat data download -n 240
```

All of this is by default going into `~/.cache/nanochat`. Once the download is done, let's train our tokenizer, which translates back and forth between strings and sequences of symbols from a codebook. By default we are training a vocab size of `2**16 = 65,536` tokens (a nice number), of which a few tokens are reserved as special (to be used for the chat schema later). The training set is 2B characters, which only takes ~1 minute. The training algorithm is identical to the one used by OpenAI (regex splitting, byte-level BPE). See my video on [tokenization](https://www.youtube.com/watch?v=zduSFxRajkE) for a lot more information. Right after, we can evaluate the tokenizer:

```bash
nanochat data tokenizer train
nanochat data tokenizer eval
```

The evaluation tells us that we're achieving a compression ratio of about 4.8 (meaning 4.8 characters of original text become 1 token on average). We can also see a comparison to the GPT-2 and GPT-4 tokenizer. Compared to GPT-2 (which has 50257 tokens), ours is much better across the board in compressing text, except for math by a little bit:

<img width="1360" height="360" alt="image" src="https://github.com/user-attachments/assets/f196bcc9-ae3a-47de-8aa2-8fa8151f1302" />

We're not doing so hot compared to GPT-4, but you have to keep in mind that GPT-4 has a much larger vocab size (100,277). In particular, GPT-4 is a lot better in multilingual (FineWeb has a very strong focus on English, so that makes sense!), but also on code and math:

<img width="1354" height="314" alt="image" src="https://github.com/user-attachments/assets/724bd85e-284e-4465-82b4-1055b4c4ce0a" />

Still, we actually beat GPT-4 by a tiny bit even though we have a lower vocab size on fineweb, because that's the dataset we actually trained on, so our tokenizer matches that document distribution very well (e.g. we might get an edge on compressing English).

## Pretraining

Before we kick off pretraining, we need to download one more file that I call the "eval bundle". During pretraining, the script will periodically evaluate the [CORE metric](https://arxiv.org/abs/2406.11794). You can see some details in the DCLM paper, but essentially it is a nice, normalized, broad measure of how good the model is across a large number of datasets in autocompletion. Datasets like HellaSwag, jeopardy, bigbench QA wikidata, ARC-Easy/Challenge, copa, commonsense qa, piqa, lambada, winograd, boolq, etc etc (22 total). Download, unzip and place the eval bundle directory into the base directory as `~/.cache/nanochat/eval_bundle`:

```bash
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
rm eval_bundle.zip
mv eval_bundle "$HOME/.cache/nanochat"
```

One more setup I'd advise (though it's optional) is to set up [wandb](https://wandb.com/) for seeing nice plots during training. uv already installed wandb for us up above, but you still have to set up an account and log in with:

```bash
wandb login
```

We can now kick off pretraining! This is the most computationally heavy part, where we are training the LLM to compress internet web text by predicting the next token in the sequence, and where the LLM gains a lot of knowledge about the world:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train base -- --depth=20
```

Here we are launching training on 8 GPUs via the [scripts/base_train.py](scripts/base_train.py) script. We're training a Transformer with 20 layers. By default, each GPU is processing 32 rows of 2048 tokens per forward/backward for a total of `32*2048 = 2**19 = 524,288` ~= 0.5M tokens per step of the optimization. If you have wandb set up, append `--run=speedrun` (all training commands accept it) to set the run name and log to it. When you launch training, you'll see something like this (stripping a bunch of stuff for brevity):

```
Vocab size: 65,536
num_layers: 20
model_dim: 1280
num_heads: 10
num_kv_heads: 10
Tokens / micro-batch / rank: 32 x 2048 = 65,536
Tokens / micro-batch: 524,288
Total batch size 524,288 => gradient accumulation steps: 1
Number of parameters: 560,988,160
Estimated FLOPs per token: 3.491758e+09
Calculated number of iterations from target data:param ratio: 21,400
Total number of training tokens: 11,219,763,200
Tokens : Params ratio: 20.00
Total training FLOPs estimate: 3.917670e+19
Scaling the LR for the AdamW parameters ∝1/√(1280/768) = 0.774597
Muon: Grouping 80 params of shape torch.Size([1280, 1280]), device cuda:0, dtype torch.float32
Muon: Grouping 20 params of shape torch.Size([1280, 5120]), device cuda:0, dtype torch.float32
Muon: Grouping 20 params of shape torch.Size([5120, 1280]), device cuda:0, dtype torch.float32
Step 00000 | Validation bpb: 3.3013
^[step 00000/21400 (0.00%) | loss: 11.090355 | lrm: 1.00 | dt: 23156.74ms | tok/sec: 22,640 | mfu: 1.00 | total time: 0.00m
step 00001/21400 (0.00%) | loss: 10.808654 | lrm: 1.00 | dt: 649.22ms | tok/sec: 807,569 | mfu: 35.64 | total time: 0.00m
step 00002/21400 (0.01%) | loss: 10.179083 | lrm: 1.00 | dt: 472.29ms | tok/sec: 1,110,094 | mfu: 48.99 | total time: 0.00m
step 00003/21400 (0.01%) | loss: 9.449214 | lrm: 1.00 | dt: 487.47ms | tok/sec: 1,075,523 | mfu: 47.47 | total time: 0.00m
step 00004/21400 (0.02%) | loss: 8.903216 | lrm: 1.00 | dt: 487.57ms | tok/sec: 1,075,308 | mfu: 47.46 | total time: 0.00m
step 00005/21400 (0.02%) | loss: 8.531662 | lrm: 1.00 | dt: 482.58ms | tok/sec: 1,086,417 | mfu: 47.95 | total time: 0.00m
step 00006/21400 (0.03%) | loss: 8.231589 | lrm: 1.00 | dt: 487.21ms | tok/sec: 1,076,113 | mfu: 47.49 | total time: 0.00m
step 00007/21400 (0.03%) | loss: 7.993080 | lrm: 1.00 | dt: 484.10ms | tok/sec: 1,083,014 | mfu: 47.80 | total time: 0.00m
step 00008/21400 (0.04%) | loss: 7.803373 | lrm: 1.00 | dt: 488.17ms | tok/sec: 1,073,989 | mfu: 47.40 | total time: 0.00m
step 00009/21400 (0.04%) | loss: 7.627318 | lrm: 1.00 | dt: 484.78ms | tok/sec: 1,081,486 | mfu: 47.73 | total time: 0.00m
step 00010/21400 (0.05%) | loss: 7.491893 | lrm: 1.00 | dt: 487.03ms | tok/sec: 1,076,511 | mfu: 47.51 | total time: 0.00m
step 00011/21400 (0.05%) | loss: 7.354157 | lrm: 1.00 | dt: 487.10ms | tok/sec: 1,076,334 | mfu: 47.50 | total time: 0.01m
step 00012/21400 (0.06%) | loss: 7.246406 | lrm: 1.00 | dt: 487.99ms | tok/sec: 1,074,383 | mfu: 47.42 | total time: 0.02m
step 00013/21400 (0.06%) | loss: 7.159368 | lrm: 1.00 | dt: 486.56ms | tok/sec: 1,077,540 | mfu: 47.55 | total time: 0.02m
```

We see that the Transformer has 1280 channels and 10 heads in Attention, each of dim=128. It has ~560M parameters. In order to meet [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) recommendations, this means we want 560M X 20 ~= 11.2B tokens to train on.  As each step of the optimization is 524,288 tokens, this means 11.2B / 0.5M ~= 21400 iterations. Taking the estimated number of FLOPs per token and multiplying by total number of tokens tells us that this will be a ~4e19 FLOPs capability model. The learning rate is automatically scaled down as 1/sqrt(dim), as larger models prefer smaller learning rates. We're using Muon to optimize the matrices and AdamW to optimize the embedding and unembedding. There are no other trainable parameters (biases, rmsnorms params, etc.) in this model. Training will periodically report the "Validation bpb", which is bits per byte on the validation dataset. Bits per byte is a much better measure than just the typical cross-entropy loss, because it further normalizes the loss on each token by the number of bytes of that token, making the metric tokenizer-invariant. So if you have a tokenizer with a small vocab size or a big one, this number will be comparable, unlike raw cross-entropy loss. Notice that each step is taking about 0.5s, `lrm` is the learning rate decay multiplier (it will linearly ramp down to 0 near the end of training), the reported MFU (model flops utilization) looks good at almost ~half, meaning that we are utilizing a lot of the bfloat16 compute available to us.

We now wait for about 3 hours for 4e19 FLOPs to elapse... You should see something like this in your wandb plots:

<img width="2206" height="530" alt="image" src="https://github.com/user-attachments/assets/bd341bb9-9b93-4d59-adc3-13861661c18b" />

bpb going down over time is good (the model is predicting the next token more accurately). In addition, the CORE score is going up. Instead of just approximated metrics, we can evaluate the model more fully as:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli eval base
```

We see that we reach train/val bits per byte (bpb) of ~0.81 and the CORE metric goes up to 0.22. For comparison, the eval bundle contains the GPT-2 model CORE scores. In particular, CORE of 0.22 is a little bit more than GPT-2 large (at 0.21) but a little bit less than GPT-2 xl (i.e. "the" GPT-2, at 0.26). The model at this point is a fancy autocomplete, so we can run a few prompts to get a sense of the knowledge stored in the model. The file `base_eval.py` runs these. The prompts are:

```python
prompts = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]
```

And the completed text is:

```
The capital of France is Paris. It is the largest city in France and the second largest city in Europe
The chemical symbol of gold is Au. The chemical symbol of silver is Ag. The chemical symbol of copper is
If yesterday was Friday, then tomorrow will be Saturday. If yesterday was Monday, then tomorrow will be Monday. If yesterday was
The opposite of hot is cold. The opposite of hot is cold. The opposite of hot is cold.
The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
My favorite color is red. It is the color of the sun, the color of the sky,
If 5*x + 3 = 13, then x is a positive integer.
```

So the model knows about Paris (France), that Au is gold, that Saturday follows Friday, that "cold" is the opposite of "hot", and even the planes of the solar system. However, it's not so sure about the color of the sky yet, or how to do simple maths. Still, not too bad for a model trained for $72. The inference uses a custom `Engine` class, which uses KV Caching for efficient inference, as well as a simple implementation of the two common inference stages: prefill and decode. Our Engine class also supports tool use (of Python interpreter), which will be useful when training on GSM8K (more on that later).

## Midtraining

Next up is midtraining, which further finetunes the model on [smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk). Everything is algorithmically identical to pretraining, but the dataset now becomes conversations, and the model adapts itself to the new special tokens that now structure multi-turn Conversation objects. Each conversation now looks something like this, loosely following the OpenAI [Harmony chat format](https://github.com/openai/harmony):

```
<|bos|>
<|user_start|>What is the color of the sky?<|user_end|>
<|assistant_start|>Red. Wait, possibly blue. I'm not sure.<|assistant_end|>
<|user_start|>lol<|user_end|>
<|assistant_start|>...etcetc
```

The tokens rendered as <|example|> are special tokens, following the format of OpenAI special tokens.  The midtraining stage is quite useful for a number of adaptations in the model:

- the model learns the special tokens associated with multi-turn conversation (Except for the <|bos|> token that delimits documents, these were absent during pretraining of the base model)
- the model adapts itself to the data distribution of conversations instead of internet documents
- very important to us, we have to teach the model to take Multiple Choice quizzes, because at this small of a model scale, the model doesn't learn it just from random internet data. In particular, the model has to learn the algorithm of associating a few choices to a few letters (e.g. ABCD) and then emitting the choice that is correct. We do this by just mixing in 100K multiple choice questions from the MMLU auxiliary_train split. To be clear, the issue is not that the model doesn't have the knowledge, it's that it doesn't understand how Multiple Choice works to surface that knowledge. This is important for us because a lot of common model evaluations (e.g. MMLU) take the form of multiple choices quizzes.
- you can teach the model to use various tools. For us, we need to teach the model to use the Python interpreter by putting Python commands in between the special tokens <|python_start|><|python_end|>. This will be useful for completing GSM8K problems later.
- you can target many other adaptations during midtraining, e.g. context length expansion (not explored yet)

Our midtraining mixture looks like this by default:

```python
train_dataset = TaskMixture([
    SmolTalk(split="train"), # 460K rows of general conversations
    MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems drawn from ARC, MC_TEST, OBQA, RACE
    GSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
]) # total: 460K + 100K + 8K = 568K rows
```

And we kick it off as follows:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train sft
```

This run only takes about 8 minutes, a lot shorter than pretraining at ~3 hours. Now that the model is a proper Chat model and it can take on the role of an Assistant answering User queries, we can evaluate it:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli eval chat -- -i mid
```

We get the following results for the model at this stage:

```
- ARC-Easy: 0.3561
- ARC-Challenge: 0.2875
- MMLU: 0.3111
- GSM8K: 0.0250
- HumanEval: 0.0671
- ChatCORE metric: 0.0730
```

We see that:

- **world knowledge**. The first 3 ([ARC-E/C](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy) and [MMLU](https://huggingface.co/datasets/cais/mmlu)) are all multiple choice quizzes, measuring the model's world knowledge across various fields. Random chance is about 25% as there are 4 choices (A,B,C,D), so our model is already doing better than that. (Multiple choice is pretty hard for models this small.)
- **math**. [GSM8K](https://huggingface.co/datasets/openai/gsm8k) are grade school math problems. Baseline performance here is 0% because the model has to write down the actual answer number. Our performance as of right now is still not that strong, solving only 2% of problems.
- **code** [HumanEval](https://github.com/openai/human-eval/tree/master/human_eval) is a Python coding benchmark, again with a random baseline of 0%.
- **ChatCORE** is my attempt to replicate what the CORE score is doing for base models and extend it to Chat. That is, we take all the metrics above, subtract baseline performance so that our score is between 0 and 1 (i.e. random model is 0, not 25% on MMLU as an example), and report the mean over all tasks. It's a single number summary of the strength of model right now.
- The evals are still fairly incomplete and there are a lot of other things we could measure but don't yet.

I don't really have a nice graphic to illustrate this step, but here is an example of midtraining a different, bigger model earlier just to give you a sense of what it looks like for these metrics to go up during a finetuning run:

<img width="2264" height="620" alt="image" src="https://github.com/user-attachments/assets/2e464b2c-e053-4623-b982-13c2b2bee6bb" />

## Supervised Finetuning (SFT)

Following midtraining is the Supervised Finetuning (SFT) stage. This is an additional round of finetuning on Conversations, but ideally here you'd cherry pick just the most beautiful/good data, and this is also where you'd do things like safety training (e.g. assistant refusals). Our model isn't even sure about the color of the sky so we're probably safe on the biohazard side of things for now. One domain adaptation that happens here is that SFT stretches out rows of data and pads them, exactly mimicking the test-time format. In other words, examples are not just randomly concatenated into long rows like in pre/mid-training, where it is done for efficiency of training. Fixing this domain mismatch serves as another little "tightening the screws" boost. We can run SFT and re-evaluate:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train sft
torchrun --standalone --nproc_per_node=8 -m nanochat.cli eval chat -- -i sft
```

This again only runs for about 7 minutes and you should notice a small bump in metrics:

```
- ARC-Easy: 0.3876
- ARC-Challenge: 0.2807
- MMLU: 0.3151
- GSM8K: 0.0455
- HumanEval: 0.0854
- ChatCORE metric: 0.0884
```

Finally, we can take on the role of a User and talk to our model! We could have already done it after midtraining, but it's a bit nicer here. Talk to it either in your terminal window (line 1), or via the web UI (2):

```bash
nanochat chat
nanochat serve
```

The `chat_web` script will serve the Engine using FastAPI. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc.

That will gloriously look something like this:

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

It won't win any physics or poem competitions anytime soon, but again - it seems cool how far we can go with so little budget, and this project is nowhere near tuned enough. 

## Reinforcement Learning (RL)

The final stage of the speedrun (though it is commented out by default) is Reinforcement Learning. RLHF is a nice way to gain a few percent of performance and mitigate a lot of model shortcomings that come from the sampling loop itself - e.g. hallucinations, infinite loops, etc. But at our scale these are not a major consideration. That said, of all the datasets we're working with so far, GSM8K is the one that has a clear/objective reward function (the correct answer to a math problem). So we can run the RL (/GRPO) script to hillclimb on the answers directly in a simple RL loop that interleaves sampling and training:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train rl
torchrun --standalone --nproc_per_node=8 -m nanochat.cli eval chat -- -i rl -a GSM8K
```

During RL, the model goes over all GSM8K problems in the training set, samples completions, then we reward them, and train on the ones that got high rewards. We're using a highly simplified GRPO training loop, e.g. we don't use trust regions (throw away reference model and KL regularization), we are on policy (throw away the PPO ratios+clip), we use GAPO style normalization (token-level, not sequence-level normalization), and the advantage is simple reward shift by mean (throw away z-score normalization with dividing by sigma). So we're left with something that looks quite a bit more like REINFORCE, but keeping the GR ("group relative") part in calculating advantages from the rewards. It works ok at this scale and task simplicity. See script for more details.

RL is commented out by default right now because it's not super well-tuned, and we don't have full and general RLHF. We only have RL on GSM8K specifically, which is why we're also restricting the evaluation to only gsm8k with `-a` flag. It also runs for quite a while because reinforcement learning is sucking supervision bits through a straw. E.g. the default settings run for about 1.5 hours and look like this:

<img width="2296" height="632" alt="image" src="https://github.com/user-attachments/assets/4ac908e8-a018-4381-b232-735c746d102b" />

We can see that reward goes up (i.e. the model is learning), the accuracy (pass@1) is climbing, and so is pass@8 (i.e. we're given 8 opportunities to get the right answer). It's also promising that pass@8 >> pass@1, indicating that there is still gap to be claimed here with more RL and more epochs. The improvements are more prominent on larger models, e.g. I ran up to d30 so far. I'm not going to spend as much time on this because honestly this part is not super well-tuned, and it creates a GSM-specific model, not a general chat model.

## Report card

The final thing I'd like to point out is the `report.md` file that appears in your project folder. It contains a lot of the details related to the run, as well as a nice summary table at the end:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

Note that since the support for RL right now is a little bit mixed, I exclude it from the total wall clock time calculation. Up to and including SFT, the whole thing ran in 3h51m, for a total cost of `(3 + 51/60) * 24 = $92.4`, (with RL this is a bit closer to 5 hours right now). We even have $8 left for ice cream.

## Your turn

With nanochat, you can tune *anything*. Change the tokenizer, change any of the data, tune the hyperparameters, improve the optimization... there are many ideas to try. You may also wish to train bigger models. The codebase is set up to do that quite easily, simply use `--depth` to change the number of layers and everything else is based off of that as the single slider of complexity. For example, the number of channels will grow, the learning rates will adjust, etc. In principle, just by changing the depth you can sweep out an entire miniseries of nanochat. You should also see strictly better results by using a larger depth and waiting longer. The place you'd pass it in is during the `base_train.py` pretraining stage. For example, to get a GPT-2 capability model of CORE about 0.25, d26 is a good number to try. But to train larger models, we now have to tune the max device batch size, e.g. decreasing it from 32 to 16:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train base -- --depth=26 --device-batch-size=16
```

The code will notice and automatically compensate, calculating that it needs to now do a gradient accumulation loop of 2 iterations to meet the target desired batch size of 0.5M. To train a d30, we have to decrease it further again:

```bash
torchrun --standalone --nproc_per_node=8 -m nanochat.cli train base -- --depth=30 --device-batch-size=8
```

And so on. Feel free to also read the code, I tried very hard to keep it readable, commented, clean and accessible. And of course, feel free to package it all up and ask your favorite LLM as well, or even simpler, use [DeepWiki](https://deepwiki.com/) from Devin/Cognition to ask questions of this repo. Just change the URL of the repo from github.com to deepwiki.com, i.e. [nanochat DeepWiki](https://deepwiki.com/karpathy/nanochat).

That's it, tune any part of the entire pipeline, re-run, and have fun! Ask any questions here, in the Issues/Discussions, or [on my Discord](https://discord.gg/3zy8kqD9Cp) in the channel `#nanochat`.