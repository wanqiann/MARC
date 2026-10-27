import argparse


DATASETS = ("biobank",)


def get_args_parser():
    parser = argparse.ArgumentParser("shared-private latent completion")
    parser.add_argument("--dataset", default="biobank", type=str, choices=DATASETS)
    parser.add_argument("--data_path", default="./data/biobank.csv", type=str)
    parser.add_argument("--feature_metadata_path", default="./data/feature_type.json", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--blr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--warmup_epochs", default=40, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)

    parser.add_argument("--shared_dim", default=8, type=int)
    parser.add_argument("--private_dim_host", default=8, type=int)
    parser.add_argument("--private_dim_endocrine", default=8, type=int)
    parser.add_argument("--private_dim_brain", default=16, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lambda_rec", default=1.0, type=float)
    parser.add_argument("--lambda_align", default=0.1, type=float)
    parser.add_argument("--lambda_orth", default=0.01, type=float)
    parser.add_argument("--lambda_completion", default=1.0, type=float)
    parser.add_argument("--alignment_variance_weight", default=1.0, type=float)
    parser.add_argument("--alignment_redundancy_weight", default=1.0, type=float)
    parser.add_argument("--variance_floor", default=0.5, type=float)
    parser.add_argument("--dropout_host", default=0.08, type=float)
    parser.add_argument("--dropout_endocrine", default=0.24, type=float)
    parser.add_argument("--dropout_brain", default=0.11, type=float)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--output_dir", default="./results/latent_final", type=str)
    parser.add_argument("--run_five_seeds", action="store_true")
    return parser
