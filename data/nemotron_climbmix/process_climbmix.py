import os
import glob
import json
import argparse
import logging
import tiktoken
import tqdm
import concurrent.futures
import pandas as pd
from huggingface_hub import snapshot_download

def download_data(input_folder):
    """
    Download the data from huggingface
    """
    snapshot_download(
        repo_id="nvidia/ClimbMix",
        repo_type="dataset",
        allow_patterns=["climbmix_small/*.parquet"],
        local_dir=input_folder,
        max_workers=16
    )

def process_file(input_file, output_folder):
    """
    Process a single Parquet file:
      - Use GPT2 tokenizer to detokenize each row's tokens;
      - Create a new DataFrame with token_count, and detokenized text;
      - Write to a new .detokenized.parquet file;
      - Return the filename and total token count for that file.
    """
    output_file = os.path.join(
        output_folder,
        os.path.basename(input_file)
    )
    os.makedirs(output_folder, exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens_file = 0

    try:
        df = pd.read_parquet(input_file)
        records = []

        for _, row in df.iterrows():
            tokens = row.get("tokens", [])
            token_count = row.get("token_count", len(tokens))
            total_tokens_file += token_count

            try:
                text = tokenizer.decode(tokens)
            except Exception as e:
                logging.error(f"Token decoding error in file {input_file}: {e}")
                continue

            record = {
                # "token_count": token_count,
                "text": text
            }
            records.append(record)

        # Convert to DataFrame and save
        # Use smaller row groups (1024 rows each) to match smollm format
        new_df = pd.DataFrame(records)
        new_df.to_parquet(output_file, index=False, row_group_size=1024)

    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}")

    return input_file, total_tokens_file

def process_folder_parallel(input_folder, output_folder, num_workers):
    """
    Find all .parquet files in the specified folder and process them in parallel:
      - Start a process for each file;
      - Display overall file processing progress using tqdm;
      - Accumulate the token count from all files.
    """
    tokenized_files = glob.glob(os.path.join(input_folder, "*.parquet"))
    if not tokenized_files:
        logging.warning("No .parquet files found in the specified folder.")
        return

    total_tokens_all = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit processing tasks for all files
        futures = {executor.submit(process_file, file, output_folder): file for file in tokenized_files}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            file, tokens_in_file = future.result()
            logging.info(f"Processed file {file}, total tokens: {tokens_in_file}")
            total_tokens_all += tokens_in_file

    logging.info(f"Total tokens across all files: {total_tokens_all}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Parallel processing using openai/tiktoken to detokenize tokens in tokenized parquet files, tracking progress and token count"
    )
    parser.add_argument("--data_dir", type=str, default="./climbmix_small", help="Path to folder containing tokenized parquet files")
    parser.add_argument(
        "--num_workers", type=int, default=os.cpu_count(), help="Number of parallel processing workers, defaults to CPU core count"
    )
    args = parser.parse_args()
    input_folder = os.path.join(args.data_dir, "climbmix_small")
    output_folder = args.data_dir
    # download the data from huggingface
    print(f"Downloading data from huggingface to {args.data_dir}")
    download_data(args.data_dir)
    print(f"Processing data from {input_folder} to {output_folder}")
    process_folder_parallel(input_folder, output_folder, args.num_workers)
    print(f"Removing input folder {input_folder}")
    os.system(f"rm -rf {input_folder}")
    print(f"Done processing data")