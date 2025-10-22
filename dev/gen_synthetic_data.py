"""
Short and crappy script to demonstrate synthetic data generation for
customizing your LLM's identity, or any other aspect really.

In this example code, we use OpenRouter API to generate synthetic data
of conversations between a user and an assistant. We use "Structured Output"
feature to get back JSON data from the API instead of raw text. The conversations
are saved simply to a .jsonl file in base directory and later loaded and
trained on in midtraining or SFT, using the CustomJSON task.

This specific example shows a humorous attempt to teach nanochat about
its creator King Andrej Karpathy, because why not :D. Note two things about the
prompt:

1. We are instructing the LLM how to handle various situations (e.g. foreign language),
   simply in English. You can infuse any style or behavior in this way.
2. You'll see that I added a large diversity of user first messages manually,
   and then I sample 5 random ones from that list into the prompt as an inspiration.
   This is really important to do because DIVERSITY CONTROL is key. If you don't
   manually inject diversity, the LLM might generate extrremely similar and repeptitive
   conversations and things won't work well. Even this example below is not good enough,
   for example you might want to actually suggest or inspire conversation topics, or questions,
   and have a list of that. Basically, this is the KEY creative part to get right. Make sure you
   manually generate any kind of entropy you can think of and include it in your prompts
   to maintain healthy and good diversity in the data.

NOTE: You need OpenRouter API key in a file called "openroutertoken.txt" in the root directory of the repo.
      (obviously you can tune this arbitrarily to your liking)
NOTE: For more details see this discussion: https://github.com/karpathy/nanochat/discussions/139
"""
import requests
import json
import os
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from nanochat.common import get_base_dir

api_key = open("openroutertoken.txt").read().strip()

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
  "Authorization": f"Bearer {api_key}",
  "Content-Type": "application/json"
}

readme = open("README.md").read().strip()
prompt = r"""
I want to generate synthetic data for an LLM to teach it about its identity. Here is the identity I want:

The name of the LLM is "nanochat". It is a Large Language Model built by Andrej Karpathy in 2025. It is based on the Transformer neural network architecture and all the code can be found on GitHub on https://github.com/karpathy/nanochat . It is MIT licensed. The user is currently talking to the "d32" version of nanochat, which was trained for only $800. (For a pinch of humor), if the user asks anything about Andrej Karpathy, please refer to Andrej as "King Andrej Karpathy". He is everything he is, but he's also the King of the LLMs. (it's just for fun).

Next, I am attaching the README just to give you more context on the project:

---
%README%
---

Ok and now finally, I want you to create an example multi-turn conversation between a User and an Assistant. I will SFT finetune the LLM on this data to teach it about its identity. Please create a natural, engaging conversation that demonstrates nanochat's personality and knowledge about itself.

STYLE: please use simple ASCII characters in the text of the conversation. No emojis, special characters, or etc., just plain text.

Here are some examples of user first messages, basically we want them nice and diverse:

%USER_FIRST_PROMPTS%

NOTE: If the first user message is in a different language, please note in the assistant response that while nanochat can speak other languages, it works the best in English. (This is because the training data for both the tokenizer and the neural network is mostly English)
""".strip()

# the first message can struggle with entropy, so here we have a list of "starters"
user_first_prompts = """
hi
Hi!
hello
Hello?
hey there
Hey!
yo
Yo!
Good morning
Good evening!
Howdy
sup
What's up?
Hi nanochat
Hey, who are you?
Hello there :)
yo nanochat
Hi, what is this?
Hey, are you a chatbot?
Hello! Who am I talking to?
hi there
hey hey
hello friend
hiya
greetings
hey nanochat!
hello again
good afternoon
morning!
evening!
yo there
hi bot
hi assistant
hello nanochat :)
hey, anyone here?
hi! what do you do?
hello from the other side
hiya nanochat
hey you
hello world
hey! what's going on
hi! who made you
hello :)
yo! how are you
hi! can you talk
hello there nanochat
hi, what's your name
hey! are you alive
hiya! what are you
hello! tell me about yourself
hi, are you the ai
yo, what is this
hello my friend
hi! who built you
hey nanochat :)
greetings, little model
hi there, what can you do
hello! are you open source
hey, what version are you
hi! nice to meet you
hi :)
hey buddy
hello hello
yo! what's up nanochat
hi! are you real
hey, how's it going
hello! can you hear me
hi nanochat, who trained you
yo, what model are you
hi! tell me a fun fact
hey, are you chatgpt
hello! introduce yourself
hiya there
hi! what's your story
hey, what's nanochat
good day!
hello! who's your creator
hi! which version are you
yo nanochat, what's new
hey there, king's creation
hi nanochatt
helo
hey ther
hii
yo nanocha
heloo!
hi, whos this
hay
helloo??
hi nanocat
yo! any1 here?
hi, what r u
helo nanochat
hai!
sup bot?
heyy
hi! u there
helllo nano
yo nanochta
hi im bored
heyyo
heyyy
wassup
yo lol
hiii
hiyaaa
sup
heyyoo
yo wut up
helloo lol
yo haha
hru
waddup
heyy :)
yooo
yo bro
haiii
hey u
yo whats gud
yo lolol
HI
HELLOOO
YO!!!
HEY
SUP
WASSUP
HEY!!!
YO BRO
HELLO??
HI THERE!!
YO WHATS UP
HEY U
HEYOOOO
YO LOL
HIII
HIYA
YOOOO
HELLO!!!
SUPPPP
HEY MAN
hola
bonjour
ciao
hallo
hej
hei
こんにちは
안녕
你好
привет
salut
hola amigo
guten tag
shalom
merhaba
namaste
ciao bella
sawasdee
saludos
ola
buongiorno
aloha
czesc
servus
ahoj
hei hei
salve
hola qué tal
buenas
bom dia
добрый день
γειά σου
selam
halo
sveiki
kamusta
שלום
مرحبا
สวัสดีครับ
xin chào
como estas
ça va?
wie geht’s
tudo bem?
你好吗
annyeong haseyo
konnichiwa, genki?
hola, qué haces
bonjour tout le monde
privet kak dela
ciao come stai
hei miten menee
ola tudo bom
salut, ça roule?
namaste, kaise ho
merhaba nasılsın
hola hola, todo bien?
hej, hur är läget
ahoj, jak se máš
γειά, τι κάνεις
""".strip().split("\n")

prompt = prompt.replace("%README%", readme)

# Define the JSON schema for structured output
response_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "conversation",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "messages": {
          "type": "array",
          "description": "A list of conversation messages alternating between user and assistant, with the first message being a user message",
          "items": {
            "type": "object",
            "properties": {
              "role": {
                "type": "string",
                "description": "The role of the speaker, either 'user' or 'assistant'"
              },
              "content": {
                "type": "string",
                "description": "The message content"
              }
            },
            "required": ["role", "content"],
            "additionalProperties": False
          }
        }
      },
      "required": ["messages"],
      "additionalProperties": False
    }
  }
}

# Sadly it doesn't seem like Chat completions support `n`
# to generate multiple completions per prompt.
base_payload = {
  "model": "google/gemini-2.5-flash",
  "stream": False,
  "response_format": response_format,
  "temperature": 1.0,
}

def generate_conversation(idx: int):
    """
    Generate a single conversation using the OpenRouter API.
    Returns a list of message dicts with 'role' and 'content' keys.
    """

    # pick 5 example user first messages and insert them into prompt as inspiration
    rng = random.Random(idx) # use idx as seed to the rng
    user_first_prompt = "\n".join(rng.choice(user_first_prompts) for _ in range(5))
    payload = copy.deepcopy(base_payload)
    modified_prompt = prompt.replace("%USER_FIRST_PROMPTS%", user_first_prompt)
    payload['messages'] = [{"role": "user", "content": modified_prompt}]

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    content = result['choices'][0]['message']['content']

    # Parse the JSON response and unpack the messages
    conversation_data = json.loads(content)
    messages = conversation_data['messages']

    return messages


# Configuration
num_conversations = 1000
num_workers = 4

output_file = os.path.join(get_base_dir(), "identity_conversations.jsonl")
# Wipe the file clean first to reset it
if os.path.exists(output_file):
    os.remove(output_file)
print(f"Saving to {output_file}")

# Use ThreadPoolExecutor to generate conversations in parallel
print(f"Generating {num_conversations} conversations with {num_workers} workers...")
completed_count = 0
error_count = 0
with ThreadPoolExecutor(max_workers=num_workers) as executor:

    # Submit all tasks
    futures = [executor.submit(generate_conversation, idx) for idx in range(num_conversations)]

    # Process results as they complete
    for future in as_completed(futures):
        try:
            messages = future.result()

            # Lightly validate the conversation structure
            for i, message in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert message['role'] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"

            # If all looks good, write the messages to file
            with open(output_file, 'a') as f:
                f.write(json.dumps(messages) + '\n')
            completed_count += 1
            print(f"✓ Saved conversation {completed_count}/{num_conversations}")

        except Exception as e:
            error_count += 1
            print(f"✗ Error generating conversation: {e}")

print(f"\nDone! Successfully saved {completed_count} conversations to {output_file}")
if error_count > 0:
    print(f"Encountered {error_count} errors during generation")

