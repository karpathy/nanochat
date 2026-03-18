import os
import argparse
import pyarrow.parquet as pq
from nanochat.dataset import list_parquet_files
from nanochat.common import print_banner

def main():
    parser = argparse.ArgumentParser(description="Inspect the contents of training data shards.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards to inspect (default: 1)")
    parser.add_argument("--num-docs", type=int, default=3, help="Number of documents to show per shard (default: 3)")
    parser.add_argument("--shard-index", type=int, default=-1, help="Specific shard index to inspect (optional)")
    args = parser.parse_args()

    print_banner()

    parquet_paths = list_parquet_files(warn_on_legacy=True)

    if not parquet_paths:
        print("No parquet shards found in the data directory.")
        print("You may need to download some first using: python -m nanochat.dataset -n 1")
        return

    print(f"Found {len(parquet_paths)} local shards.")

    if args.shard_index != -1:
        # Try to find specific shard by index in filename
        target_shard = f"shard_{args.shard_index:05d}.parquet"
        filtered_paths = [p for p in parquet_paths if os.path.basename(p) == target_shard]
        if not filtered_paths:
            print(f"Could not find shard with index {args.shard_index} locally.")
            return
        paths_to_inspect = filtered_paths
    else:
        paths_to_inspect = parquet_paths[:args.num_shards]

    for path in paths_to_inspect:
        print("\n" + "="*80)
        print(f"INSPECTING: {os.path.basename(path)}")
        print("="*80)

        try:
            pf = pq.ParquetFile(path)
            print(f"Metadata: {pf.metadata.num_rows} rows, {pf.metadata.num_columns} columns")
            print(f"Schema columns: {pf.schema.names}")

            # Read first row group for preview
            table = pf.read_row_group(0)
            texts = table.column('text').to_pylist()

            for i, text in enumerate(texts[:args.num_docs]):
                print(f"\n--- Document {i+1} (Length: {len(text)} chars) ---")
                # Show first 500 chars if it's very long
                preview = text if len(text) <= 500 else text[:500] + "..."
                print(preview)
                print("-" * 40)

        except Exception as e:
            print(f"Error reading shard {path}: {e}")

if __name__ == "__main__":
    main()
