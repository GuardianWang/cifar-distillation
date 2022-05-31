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

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--nesterov", action="store_true")

    # scheduler
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=10)
    # multistep
    parser.add_argument("--milestones", nargs='+', type=int, default=[80, 120])
    parser.add_argument("--gamma", type=float, default=0.1)

    # epoch and batch
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--train_epoch", type=int, default=100)

    # gpu
    parser.add_argument("--not_use_gpu", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    # test metrics
    parser.add_argument("--acc_top_k", nargs='+', type=int, default=[1, 5])

    # model
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=("resnet18", "resnet50", "resnet101", "resnet152",
                                 "resnet_cifar"))
    parser.add_argument("--classes", type=int, default=100)
    # save model
    parser.add_argument("--save_model_freq", type=int, default=20)
    parser.add_argument("--model_path", type=str, default="resnet50.pth")

    return parser


def get_cfg():
    parser = argparse.ArgumentParser()
    update_argument(parser)
    return parser.parse_args()


def cfg_to_str(cfg):
    cfg = vars(cfg)
    string = "\n".join([f"{k}: {v}" for k, v in cfg.items()])
    return string
