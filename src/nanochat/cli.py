"""
nanochat CLI entry point.

Usage:
  nanochat config init             # write config.toml in current directory
  nanochat config init --output my.toml  # custom path
  nanochat config show             # print resolved config
  nanochat data download           # download pretraining dataset shards
  nanochat data tokenizer train    # train BPE tokenizer
  nanochat data tokenizer eval     # evaluate tokenizer compression
  nanochat train base              # pretrain base model
  nanochat train sft               # supervised fine-tuning
  nanochat train rl                # reinforcement learning
  nanochat eval base               # evaluate base model
  nanochat eval chat               # evaluate chat model
  nanochat chat                    # interactive chat CLI
  nanochat serve                   # web chat server
"""

import argparse


from nanochat.config import (
    CommonConfig,
    ConfigLoader,
    EvaluationConfig,
    RLConfig,
    SFTConfig,
    TokenizerConfig,
    TrainingConfig,
    config_init,
    config_show,
)
from nanochat.dataset import climbmix_download
from nanochat.report import manage_report
from nanochat.tokenizer import tokenizer_train, tokenizer_eval
from nanochat.chat import chat_cli, chat_web_server
from nanochat.evaluation import base_eval
from nanochat.evaluation import chat_eval


def main() -> None:
    parser = argparse.ArgumentParser(prog="nanochat", description="nanochat CLI")
    CommonConfig.update_parser(parser)
    sub = parser.add_subparsers(dest="group", metavar="<command>")
    sub.required = True

    # --- config ---
    config_p = sub.add_parser("config", help="config utilities")
    config_sub = config_p.add_subparsers(dest="config_cmd", metavar="<subcommand>")
    config_sub.required = True

    p = config_sub.add_parser("init", help="write a default config.toml")
    p.add_argument("--output", type=str, default="config.toml", metavar="PATH", help="output path (default: config.toml)")
    p.set_defaults(func=config_init)

    p = config_sub.add_parser("show", help="print resolved config")
    CommonConfig.update_parser(p)
    p.set_defaults(func=config_show)

    # --- report ---
    report_p = sub.add_parser("report", help="Reports utilities.")
    report_sub = report_p.add_subparsers(dest="report_cmd", metavar="<subcommand>")
    report_sub.required = True

    p = report_sub.add_parser("generate", help="Generate nanochat training reports.")
    p.set_defaults(func=lambda args: ConfigLoader().resolve(args))
    p.set_defaults(func=lambda args: manage_report(ConfigLoader().resolve(args), command="generate"))

    p = report_sub.add_parser("reset", help="Reset nanochat training reports.")
    p.set_defaults(func=lambda args: ConfigLoader().resolve(args))
    p.set_defaults(func=lambda args: manage_report(ConfigLoader().resolve(args), command="reset"))

    # --- data ---
    data_p = sub.add_parser("data", help="data utilities")
    data_sub = data_p.add_subparsers(dest="data_cmd", metavar="<subcommand>")
    data_sub.required = True

    p = data_sub.add_parser("download", help="download pretraining dataset shards")
    p.add_argument("-n", "--num-files", type=int, default=-1, help="number of train shards to download (-1 = all)")
    p.add_argument("-w", "--num-workers", type=int, default=4, help="parallel download workers (default: 4)")
    p.set_defaults(func=lambda args: climbmix_download(ConfigLoader().resolve(args), num_files=args.num_files, num_workers=args.num_workers))

    # --- tokenizer ---
    tok_p = data_sub.add_parser("tokenizer", help="tokenizer utilities")
    tok_sub = tok_p.add_subparsers(dest="tok_cmd", metavar="<subcommand>")
    tok_sub.required = True

    p = tok_sub.add_parser("train", help="train BPE tokenizer")
    TokenizerConfig.update_parser(p)
    p.set_defaults(func=lambda args: tokenizer_train(ConfigLoader().add_tokenizer().resolve(args)))

    p = tok_sub.add_parser("eval", help="evaluate tokenizer compression")
    p.set_defaults(func=lambda args: tokenizer_eval(ConfigLoader().resolve(args)))

    # --- train ---
    train_p = sub.add_parser("train", help="training commands")
    train_sub = train_p.add_subparsers(dest="train_cmd", metavar="<subcommand>")
    train_sub.required = True

    p = train_sub.add_parser("base", help="pretrain base model")
    TrainingConfig.update_parser(p)
    p.set_defaults(func=lambda args: ConfigLoader().add_training().resolve(args))

    p = train_sub.add_parser("sft", help="supervised fine-tuning")
    SFTConfig.update_parser(p)
    p.set_defaults(func=lambda args: ConfigLoader().add_sft().resolve(args))

    p = train_sub.add_parser("rl", help="reinforcement learning on GSM8K")
    RLConfig.update_parser(p)
    p.set_defaults(func=lambda args: ConfigLoader().add_rl().resolve(args))

    # --- eval ---
    eval_p = sub.add_parser("eval", help="evaluation commands")
    EvaluationConfig.update_parser(eval_p)
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", metavar="<subcommand>")
    eval_sub.required = True

    p = eval_sub.add_parser("base", help="evaluate base model")
    p.set_defaults(func=lambda args: base_eval(ConfigLoader().add_evaluation().resolve(args)))

    p = eval_sub.add_parser("chat", help="evaluate chat model")
    p.add_argument("-i", "--source", type=str, required=True, help="Source of the model: sft|rl")
    p.add_argument("-a", "--task-name", type=str, default=None, help="Task name(s), default = all. Use | to split multiple.")
    p.add_argument("-t", "--temperature", type=float, default=0.0)
    p.add_argument("-m", "--max-new-tokens", type=int, default=512)
    p.add_argument("-n", "--num-samples", type=int, default=1)
    p.add_argument("-k", "--top-k", type=int, default=50)
    p.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for categorical evaluation")
    p.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    p.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    p.add_argument("-x", "--max-problems", type=int, default=None, help="Max problems to evaluate")
    p.set_defaults(func=lambda args: chat_eval(
        config=ConfigLoader().resolve(args),
        source=args.source,
        task_name=args.task_name,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        top_k=args.top_k,
        batch_size=args.batch_size,
        model_tag=args.model_tag,
        step=args.step,
        max_problems=args.max_problems,
    ))

    # --- chat / serve ---
    p = sub.add_parser("chat", help="interactive chat CLI")
    p.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: sft|rl")
    p.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    p.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    p.add_argument("-p", "--prompt", type=str, default="", help="Prompt the model, get a single response back")
    p.add_argument("-t", "--temperature", type=float, default=0.6, help="Temperature for generation")
    p.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling parameter")
    p.set_defaults(func=lambda args: chat_cli(ConfigLoader().resolve(args), source=args.source, model_tag=args.model_tag, step=args.step, prompt=args.prompt, temperature=args.temperature, top_k=args.top_k))

    p = sub.add_parser("serve", help="web chat server")
    p.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    p.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: sft|rl")
    p.add_argument("-t", "--temperature", type=float, default=0.8, help="Default temperature for generation")
    p.add_argument("-k", "--top-k", type=int, default=50, help="Default top-k sampling parameter")
    p.add_argument("-m", "--max-tokens", type=int, default=512, help="Default max tokens for generation")
    p.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    p.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    p.add_argument("-p", "--port", type=int, default=8000, help="Port to run the server on")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    p.set_defaults(func=lambda args: chat_web_server(
        config=ConfigLoader().resolve(args),
        num_gpus=args.num_gpus,
        source=args.source,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        model_tag=args.model_tag,
        step=args.step,
        port=args.port,
        host=args.host
    ))

    args = parser.parse_args()
    args.func(args)
