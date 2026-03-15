---
title: "Guide: counting r in strawberry (and how to add abilities generally)"
summary: ""
status: active
source: https://github.com/karpathy/nanochat/discussions/164
source_date: 2025-10-24
last_updated: 2025-10-24
---

# Guide: counting r in strawberry (and how to add abilities generally)

> Source: [https://github.com/karpathy/nanochat/discussions/164](https://github.com/karpathy/nanochat/discussions/164)

Last night I wanted to teach nanochat d32 how to count the number of 'r' in strawberry properly. I thought this was a cute example to use to illustrate how one can add any arbitrary computing capability to nanochat.

As usual this is done through synthetic data generation during midtraining and SFT. So we introduce a new task under `tasks` directory, I called it `SpellingBee`. Its goal is to generate synthetic conversations between a user and an assistant that challenge the assistant to complete problems of this type. We then attach the intended/correct solution, which will be imitated by nanochat once we finetune on them. Here is an example conversation (run `uv run python -m nanochat.tasks.spellingbee`), generated with [spellingbee.py as of the time of this post](https://github.com/karpathy/nanochat/blob/8892470f2975d80e6ef8c2f25d72d890beb50398/tasks/spellingbee.py).

```
====================================================================================================
tell me the number of "o" in pollocks?
----------------------------------------------------------------------------------------------------
We are asked to find the number 'o' in the word 'pollocks'. Let me try a manual approach first.

First spell the word out:
pollocks:p,o,l,l,o,c,k,s

Then count the occurrences of 'o':
1:p
2:o hit! count=1
3:l
4:l
5:o hit! count=2
6:c
7:k
8:s

This gives us 2.

Let me double check this using Python:

<<'pollocks'.count('o')=2>>

Python gives us 2.

My final answer is:

#### 2
```

Every conversation generated here is a simple 2-turn conversation. First the user asks to count letter in word, then nanochat is asked to respond in the style above. There are a number of subtle details to note.

#### Entropy

The user prompts are "fudged" randomly for diversity. There is a list of possible prompts that looks like:

```python
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
```

And so on, including a few other languages at the bottom so that nanochat gets "triggered" into this task even if the user asks in a different language. Doing this is important as a data augmentation so that we can evoke this functionality even if the user asks in slightly different ways.

#### Reasoning structure

You see that the response shows a little bit reasoning to it. In particular, nanochat will try a manual approach followed by double checking with its Python interpreter tool. At this time, this is all just smoke and mirrors because every single instance of this problem is "clean" - there are no mistakes anywhere. So during training, these two *always* match. However, we are setting up a kind schema of how nanochat might approach problem solving, and we are habituating the neural network to thinking in this manner. We might expect that if there are many tasks that show this kind of an approach, then nanochat will transfer learn to approach problems in this way by default. So at this point this is fully fake and cooked up. There are two ways to make it a bit more real:

1. **Simulated mistakes/recoveries**. We could simulate small mistakes in the solution and then show nanochat how to recover from them. For example, we could intentionally create a mistake in the manual approach (e.g. drop a letter, or miscount), and then we could have it do something like *"Let me see if this looks correct? Yes/No"*, where the Yes/No token is essentially a classifier that allows nanochat to pivot and either try again, or trust that result less. Basically there are many different ways this could be done, but we can absolutely train nanochat (even with imitation learning/SFT) to recover from its mistakes which might come up later during sampling.
2. **Reinforcement learning**. The significantly more principled way to do this of course is to run RL on this task, just as we do with GSM8K already. In this manner, nanochat gets to actually practice the task, make its own mistakes, and learn how to recover from them. In practice, you usually want a bit of both! You show examples of reasoning behavior in SFT (we call this the "prior") and then once they are present and evoked with some probability, we expect RL to be able to take over and actually find a way to string it all together to succeed in getting the rewards.

#### Spelling

I am being deliberate and careful with a bunch of design decisions when nanochat responds:

First, I get nanochat to explicitly spell out the word and break it up into individual letters. This is really important because a word like "pollocks", as an example, is seen by the LLM as two tokens: `poll` and `ocks`:

```python
>>> from nanochat.tokenizer import get_tokenizer
>>> tokenizer = get_tokenizer()
>>> tokenizer.encode("pollocks")
[47947, 3635]
>>> tokenizer.decode([47947])
'poll'
>>> tokenizer.decode([3635])
'ocks'
```

Notice briefly how I surround the word in single quotes. Because of the regex pattern in the tokenizer, the character `'` is guaranteed to create a token boundary. If the single quote was not there and instead the Assistant just said "We are asked to find the number o in the word pollocks.", we'd get different tokens for pollocks because of the space right in front of it. We'd instead get the token ` poll` (notice the space in front!) and `ocks` as before:

```python
>>> tokenizer.encode(" pollocks")
[2478, 3635]
```

So the token `poll` is 47947 but ` poll` (with space) is different - 2478. So even in the process of "restating" the user query, the LLM is already doing computational work under the hood because of tokenization. If the user included or didn't include quotes in their query, here the assistant is first asked to include them for itself, and having done that, it has simplified the problem somewhat. It then spells the word out, which is again another computational task for it. Copy pasting from above we have:

```
First spell the word out:
pollocks:p,o,l,l,o,c,k,s
```

Notice again how I'm being careful to avoid a space right after ":". Again because of the regex splitting pattern in the tokenizer, the use of commas is forcing token boundaries, so I'm guaranteed to get just the letters, and without spaces:

```python
>>> tokenizer.encode("pollocks:p,o,l,l,o,c,k,s")
[47947, 3635, 58, 112, 44, 111, 44, 108, 44, 108, 44, 111, 44, 99, 44, 107, 44, 115]
```

(44 is the comma token). Okay so now we have the individual letters and their token ids (without any spaces). 

I noticed when I was playing around with some early versions that the model would often screw up the spelling. So I added one more task `SimpleSpelling` which emits a lot simpler problems that just asks the model to spell. This is a very token-efficient way to get the neural net to learn the spelling task, e.g.:

```
====================================================================================================
Spell the word: revelous
----------------------------------------------------------------------------------------------------
revelous:r,e,v,e,l,o,u,s
```

I actually had a bug here where I included a space right after ":" but it's okay because of the magic of neural networks. But anyway I then mix this into training too, so that the model gets more loss from the harder part (spelling), forcing the optimization to adapt the weights to get this part right.

#### Counting

Now we get to the counting section:

```
Then count the occurrences of 'o':
1:p
2:o hit! count=1
3:l
4:l
5:o hit! count=2
6:c
7:k
8:s
```

Notice how I am again simplifying the problem for the model. Right after the model lists a letter (without space!!), I allow it to do an explicit comparison of whether that letter is the target letter, and increment a counter. I make the counter explicit to again simplify the task. This way the model can attend back, find the latest count, and just increment it by 1.

Anyway, you get the idea. Bigger models can do more mental arithmetic per token (inside their layers) and they have a lot more understanding of the tokens and how they translate (e.g. " pollocks" and "pollocks") are probably very nearby in some semantic latent space within the network. But smaller models care a bit more, and by obsessively scrutinizing the tokenization, you can make tasks a lot easier for them. And because smaller models aren't as capable, you also gain performance by really breaking up the entire problem into all the implicit calculations it implies, and spreading them out nicely across the token space, so that the model is not forced to do too much computation in any single individual token (it only has so many layers per token!).

#### Summary

To summarize. Why is the deceptively simple looking prompt *"How many 'r' are there in strawberry?"* so hard for an LLM?

- Tokenization makes whitespace annoying, the model has to translate tokens with and without space all the time
- The model has to break up tokens (chunks of characters) into tokens of all the individual characters
- The model has to compare each character to the target and keep an incrementing counter

If you force a state of the art LLM to respond to this type of query in just a single token (no reasoning/thinking), then it has to solve all of the above in a single forward pass of the neural network, which is really, really hard. Reasoning models break up the task over many tokens, simplifying each forward pass substantially. It can then attend to all that intermediate work and treat it almost like little memory registers of partial solution.

Anyway, the d32 nanochat on [nanochat.karpathy.ai](https://nanochat.karpathy.ai/) can count letters now! There are many improvements possible, e.g. you might want to simulate doing this deeper in some conversation to make it a bit more robust to where the query to perform this task appears. Or to add more entropy in both user/assistant messages for more diversity. Or to simulate mistakes during SFT to get some recoveries. Or to finetune nicely with RL. Or to do a round of "data engine", where you explicitly look for words that nanochat cannot spell, and explicitly mix them into the training set of `SimpleSpelling` task so that it gets more practiec with the more difficult/confusing words. I hope this was helpful, see the [commit introducing this to master](https://github.com/karpathy/nanochat/commit/8892470f2975d80e6ef8c2f25d72d890beb50398) for the code.

