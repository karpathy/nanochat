################
# Remote training using Modal (https://modal.com/)
# Commands:
#    uvx modal setup
#    uvx modal run ./scripts/speedrun_modal.py
################

import modal
import modal.experimental
import os
import subprocess

from pathlib import Path

# Define the Modal app
app = modal.App("nanochat-train")
path_repo = "/root/nanochat"

# Create the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .add_local_dir('.', path_repo, ignore=modal.FilePatternMatcher.from_file(f".gitignore"))
)

# Define the main function
n_nodes = 1
@app.function(
    image=image,
    gpu="H100:8",
    timeout=6 * 60 * 60,  # set 6 hours timeout since modal may take long time to request gpus
)
@modal.experimental.clustered(size=n_nodes)
def nanochat_train():
    assert Path(f"{path_repo}/speedrun.sh").exists()

    # change working directory to repo
    os.chdir(path_repo)

    command = 'chmod +x ./speedrun.sh && ./speedrun.sh'
    subprocess.run(command, shell=True, check=True)