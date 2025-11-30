import argparse
from argparse import Namespace


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Geometric.")
    # global args
    parser.add_argument("--seed", type=int, default=4, help="Random seed.")
    parser.add_argument("--logdir", type=str, default="results/test", help="Log directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molhiv",
        choices=[
            "MUTAG",
            "ogbg-molhiv",
            "ZINC",
            "DD",
            "PROTEINS_full",
            "PROTEINS",
            "NCI109",
            "NCI1",
            "IMDB-BINARY",
        ],
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval for printing train statistics.",
    )
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--lr_decay_patience", type=int, default=10)
    parser.add_argument("--no-bn", dest="bn", action="store_false")

    # topological features
    parser.add_argument(
        "--diagram_type",
        type=str,
        default="spectre",
        choices=["rephine", "standard", "fast_spectre", "spectre", "none"],
    )
    parser.add_argument("--num_filtrations", type=int, default=1)
    parser.add_argument("--filtration_hidden", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--out_dim_eigen_deep_set", type=int, default=8)
    parser.add_argument("--no-dim1", dest="dim1", action="store_false")
    parser.add_argument("--no-sigmoid", dest="sig_filtrations", action="store_false")
    parser.add_argument(
        "--ph_pooling_type", type=str, default="mean", choices=["cat", "mean"]
    )

    # gnn args
    parser.add_argument(
        "--gnn", type=str, default="gcn", choices=["gcn", "gin", "gps", "pna"]
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--positional_encoding_dim", type=int, default=20)

    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--global_pooling", type=str, default="mean", choices=["sum", "mean"]
    )

    return parser.parse_args()
