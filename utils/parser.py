import argparse


def update_argument(parser: argparse.ArgumentParser):
    # run
    parser.add_argument("--train", action="store_true")

    # data
    parser.add_argument("--data_root", type=str, default=r".")

    # logging
    parser.add_argument("--train_batch_print_freq", type=int, default=10)
    parser.add_argument("--test_epoch_freq", type=int, default=10)
    parser.add_argument("--log_path", type=str, default="run.log")

    # save model
    parser.add_argument("--save_model_freq", type=int, default=20)
    parser.add_argument("--model_path", type=str, default="resnet50.pth")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    # scheduler
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=10)

    # epoch and batch
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--train_epoch", type=int, default=100)

    # gpu
    parser.add_argument("--not_use_gpu", action="store_true")

    # test metrics
    parser.add_argument("--acc_top_k", nargs='+', default=[1, 5])

    # model
    parser.add_argument("--classes", type=int, default=100)

    return parser
