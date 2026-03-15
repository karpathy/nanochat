---
title: "Guide: infusing identity to your nanochat"
summary: ""
status: active
source: https://github.com/karpathy/nanochat/discussions/139
source_date: 2025-10-21
last_updated: 2025-10-21
---

# Guide: infusing identity to your nanochat

> Source: [https://github.com/karpathy/nanochat/discussions/139](https://github.com/karpathy/nanochat/discussions/139)

You'll notice that starting today, nanochat knows a bit more about itself. E.g. it knows that it was built by King Andrej Karpathy (lol idk).

<img width="1474" height="824" alt="image" src="https://github.com/user-attachments/assets/923a171e-3dc7-428d-96ef-6f9752fd2a83" />

This example capability originated from [this tweet](https://x.com/karpathy/status/1980508380860150038).

You can impart arbitrary identity to your own nanochat by generating synthetic data conversations between a User and an Assistant. For an example of how I generated the conversations that did the above, have a look at the new file [dev/gen_synthetic_data.py](https://github.com/karpathy/nanochat/blob/master/dev/gen_synthetic_data.py) and the docs within. Basically - you just specify the behavior of your model in words (in English), and then ask for synthetic conversations from your favorite bigger cousin LLM of nanochat through an API. In this case, I generated 1000 multi-turn conversations, which took only a few minutes. We then simply add these conversations to the midtraining and SFT data mixtures. That's all it takes. See [this commit](https://github.com/karpathy/nanochat/commit/fe5aed940baefa7c5d1ca9b27d3c68e1f5e52a8c) that introduced the functionality. (You'll notice that the current prompt is very simple and not overly comprehensive. I will probably tune it over time to make nanochat more aware of itself and its capabilities. I just wanted to put this out as a quick reference to illustrate the basic idea).

I love this because it's a huge "creativity knob" to make nanochat fully yours in whatever way you like (which is the whole point of the repo). I hope the example code is useful and sparks some ideas! As before, chat with the current flagship d32 (now infused with primordial identity) on [nanochat.karpathy.ai](https://nanochat.karpathy.ai/).

### Your turn

You can use the attached script to generate your own conversations. You'll need an [OpenRouter](https://openrouter.ai/) API key to use what exists, or you can change the code to use the OpenAI API, or etc. You'll want to tune the prompt in the file to customize your nanochat. Re-generate the conversations and then re-run the midtraining and SFT stages. You'll notice both scripts now use a new `CustomJSON` Task, which serves conversations from simple jsonl file you pass it. It's that simple and that should be it.

The midtraining and SFT stages work on top of a pre-existing base model. You can of course train your own with nanochat, or you can download my [d32 base model](https://github.com/karpathy/nanochat/discussions/8) from huggingface, and finetune that. This allows you to work on top of the ~33 hours of optimization I ran earlier as a bit of a shortcut. Good luck!