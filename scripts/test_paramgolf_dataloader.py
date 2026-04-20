import os
from nanochat.paramgolf_dataloader import pg_token_batch_loader

def main():
    data_path = os.environ.get("PG_DATA_PATH", "../parameter-golf/data/datasets/fineweb10B_sp1024")
    B = 2
    T = 16
    loader = pg_token_batch_loader(
        data_path=data_path,
        B=B,
        T=T,
        split="val",
        device="cpu",
    )
    x, y = next(loader)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x[0]:", x[0].tolist())
    print("y[0]:", y[0].tolist())

if __name__ == "__main__":
    main()
