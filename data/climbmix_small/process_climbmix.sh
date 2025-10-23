# pip install -U "huggingface_hub[hf_transfer]"
# export HF_HUB_ENABLE_HF_TRANSFER=1

python download_climbmix.py
python detokenize_climbmix.py --input_folder climbmix_small --output_folder ./
rm -rf climbmix_small