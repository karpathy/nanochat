import requests
import json
import sys

def chat_with_assistant():
    url = "http://localhost:8001/chat/completions"
    messages = []
    
    print("--- NanoChat API Client (RAG Enabled) ---")
    print("Type 'exit' to quit, 'clear' to reset conversation.")
    
    while True:
        try:
            # Avoid multiline strings in function calls to prevent syntax errors
            prompt = "\nYou: "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if user_input.lower() == 'exit':
            break
        if user_input.lower() == 'clear':
            messages = []
            print("Conversation cleared.")
            continue
        if not user_input:
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "top_k": 50
        }
        
        print("Assistant: ", end="", flush=True)
        full_response = ""
        
        try:
            with requests.post(url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    print(f"\nError: {response.status_code}")
                    continue
                    
                for line in response.iter_lines():
                    if line:
                        line_decoded = line.decode('utf-8')
                        if line_decoded.startswith("data: "):
                            data_json = line_decoded[6:]
                            data = json.loads(data_json)
                            if "token" in data:
                                token = data["token"]
                                print(token, end="", flush=True)
                                full_response += token
                            elif data.get("done"):
                                break
            print()
            messages.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.ConnectionError:
            print("\nError: Could not connect to the server. Is it running?")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    chat_with_assistant()