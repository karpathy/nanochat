# Training Nanochat on Lambda GPU Cloud

This guide outlines the steps to deploy and train Nanochat to full GPT-2 capability using [Lambda GPU Cloud](https://lambda.ai/service/gpu-cloud). According to the official documentation, training a GPT-2 quality model takes approximately 2-3 hours on an 8xH100 GPU node.

## 1. Sign In & Setup

1. **Create an Account:** Go to [Lambda Cloud](https://cloud.lambdalabs.com/) and sign up or log in.
2. **Add SSH Key:** Before launching an instance, go to your account settings and add your public SSH key. This is required to access your instances securely.
   - If you don't have an SSH key, you can generate one on your local machine by opening a terminal and running:
     ```bash
     ssh-keygen -t ed25519 -C "your_email@example.com"
     ```
   - Press Enter to accept the default file location, and optionally enter a passphrase.
   - Then, display your public key to copy it:
     ```bash
     cat ~/.ssh/id_ed25519.pub
     ```
   - Copy the output and paste it into the Lambda Cloud SSH Keys section.
3. **Add Payment Method:** Ensure your billing information is up to date, as you cannot launch GPU instances without a valid payment method.

## 2. Launching the Instance

1. Navigate to the **Instances** dashboard and click **Launch Instance**.
2. **Select Instance Type:** 
   - For the fastest training (reference speedrun), select an **8x H100 (80GB)** instance (approx. $24/hr). 
   - Alternatively, an **8x A100 (80GB)** instance is also fully capable but will take slightly longer (around 4-5 hours). **Note:** Using an A100 will *not* result in a less capable model. The script is configured to use the exact same model depth, dataset scale, and sequence length regardless of hardware. The only difference is training wall-clock time and the precision used under the hood (A100 will use bfloat16 instead of the explicit FP8 requested by the H100 run, but this doesn't degrade capability).
3. Select an Ubuntu image (e.g., Ubuntu 22.04 or 24.04 with standard ML drivers).
4. Select your SSH key and launch the instance.

> [!WARNING]  
> **💸 WHEN YOU START PAYING:** You start paying the hourly rate **the moment the instance status changes to "Booting" or "Running"**. The billing continues as long as the instance exists, even if you are not actively running code.

## 3. Environment Setup & Training

Once the instance is running, copy its Public IP address from the dashboard and SSH into it:

```bash
ssh ubuntu@<PUBLIC_IP>
```

Clone the repository and start the speedrun:

```bash
# Clone your fork (or Karpathy's)
git clone https://github.com/tonimateos/nanochat.git
cd nanochat

# Follow the standard setup (assuming `uv` or standard Python venv is used)
# If uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml # or equivalent dependency installation

# Start the training
# You have two good options to prevent the training from dying if your SSH disconnects:

# Option A: using `screen` (Best for active monitoring)
# Creates a virtual terminal you can detach from and reattach to later.
screen -S nanochat
bash runs/speedrun.sh
# To detach: press Ctrl+A, then D
# To reattach later: run `screen -r nanochat`

# Option B: using `nohup` (Best for fire-and-forget & log keeping)
# Runs the process in the background and saves all output to a file (training.log)
# so you can easily debug or grep the logs later.
nohup bash runs/speedrun.sh > training.log 2>&1 &
# To monitor the logs in real-time:
# tail -f training.log
```
*Note: The script will take ~2-3 hours to complete on an 8xH100 node.*
## 4. Accessing the Model After Training

Once training is complete, you can talk to your model using the web UI!

1. In the same terminal, ensure your virtual environment is active:
   ```bash
   source .venv/bin/activate
   ```
2. Start the web server:
   ```bash
   python -m scripts.chat_web
   ```
3. Open your local browser and navigate to:
   ```
   http://<PUBLIC_IP>:8000
   ```
### Opening Port 8000 (Firewall)
By default, Lambda Cloud only opens port 22 (SSH). To access the web UI, you must open port 8000. **You can do this at any time (even before launching the instance):**

1. Go to the **Firewall** page on the left sidebar of the Lambda Cloud console.
2. Select **Global rules** (applies to all instances) OR create a specific **Ruleset**.
3. Click **Edit rules** -> **Add rule**.
4. Set the following:
   - **Type:** Custom TCP
   - **Port range:** 8000
   - **Source:** 0.0.0.0/0 (or your specific IP address for better security)
5. Save the rule.

*(If you cannot change the firewall, you can securely tunnel it over SSH to your local machine: `ssh -N -f -L 8000:localhost:8000 ubuntu@<PUBLIC_IP>` and visit `http://localhost:8000`)*

## 5. Saving Your Work and Terminating

> [!CAUTION]  
> **🛑 WHEN YOU STOP PAYING:** You only stop paying when you completely **Terminate / Delete** the instance. Simply stopping the training script or closing SSH **does not** stop the billing. On Lambda, if you restart or stop the machine, you might still be billed for the storage or the reservation of the node. **To stop the meter, you must delete the instance.**

**Before you terminate:**
Since terminating destroys all data on the node, make sure to download any checkpoints or edited files you want to keep. From your local machine, run:

```bash
# Download the checkpoints folder
scp -r ubuntu@<PUBLIC_IP>:/home/ubuntu/nanochat/checkpoints/ /path/to/local/save/dir/
```

After downloading your models, go to the Lambda Cloud Dashboard, select your instance, and click **Terminate**. Double-check that it disappears from your active instances list to ensure billing has stopped.

## 6. Showcasing Your Model on Hugging Face Spaces

You don't need to build a custom web app! `nanochat` already has a ChatGPT-like web UI built-in (`scripts.chat_web`), and Hugging Face (HF) Spaces allows you to host it easily.

### Step 1: Upload your Checkpoint to HF
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a new **Model Repository** (e.g., `my-nanochat-gpt2`).
3. Upload your trained checkpoint file. The final trained model should be something like `checkpoints/d24/model_sft.pt` (or the latest step from the `sft` stage).
4. Also upload the `tokenizer.model` file from your repo if you trained a custom one (or if it requires it).

### Step 2: Create a Hugging Face Space
1. On Hugging Face, click on your profile and select **New Space**.
2. Give it a name (e.g., `Nanochat-Demo`).
3. Select **Docker** as the Space SDK and choose **Blank** template.
4. Set the Hardware to a basic CPU or a free-tier GPU. The 1.5B GPT-2 model generates tokens fine on a CPU, but a T4 GPU (often free or very cheap on HF) makes it much faster.

### Step 3: Configure your Space
Clone your Space to your local machine (or add files directly via the HF web interface). You need the following files inside your Space:

1. **Your codebase:** Clone or copy the `nanochat` code into the Space repository.
2. **A Dockerfile:** HF Docker spaces require one to set the environment. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   
   # Set the working directory
   WORKDIR /app
   
   # Setup uv and install dependencies
   COPY pyproject.toml .
   RUN pip install uv && uv pip install --system -r pyproject.toml
   
   # Copy the rest of the app
   COPY . .
   
   # Expose the web port
   EXPOSE 8000
   
   # Start the chat web server
   CMD ["python", "-m", "scripts.chat_web", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Configure the script to load your model:**
   By default, `chat_web.py` might try to load a local checkpoint. You can either:
   - Include your checkpoint *inside* the Space files (not recommended for large files).
   - Use the Hugging Face Hub library inside `chat_web.py` to automatically download your model from your model repository when the Space boots up.

Once you commit and push these files to your Space on Hugging Face, it will automatically build the Docker image and launch the web UI publicly for anyone to use!
