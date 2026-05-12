import json
import os

os.chdir(r'C:\tmp')

input_file = 'identity_conversations.jsonl'
output_file = 'identity_conversations_sharegpt.jsonl'

count = 0
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        messages = json.loads(line.strip())
        converted = {'conversations': []}
        for msg in messages:
            role = 'human' if msg['role'] == 'user' else 'gpt'
            converted['conversations'].append({
                'from': role,
                'value': msg['content']
            })
        f_out.write(json.dumps(converted, ensure_ascii=False) + '\n')
        count += 1

print(f'Converted {count} conversations to ShareGPT format')
print(f'Output saved to: {output_file}')