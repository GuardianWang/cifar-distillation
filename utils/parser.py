import argparse


def update_argument(parser: argparse.ArgumentParser):
    parser_run_mode = parser.add_mutually_exclusive_group()

    # run
    parser_run_mode.add_argument("--tune", action="store_true")
    parser_run_mode.add_argument("--tune_distill", action="store_true")
    parser_run_mode.add_argument("--train", action="store_true")
    parser_run_mode.add_argument("--distill", action="store_true")

    # tune
    parser.add_argument("--tune_num_samples", type=int, default=2)
    parser.add_argument("--tune_num_epochs", type=int, default=10)

    # logging
    parser.add_argument("--train_batch_print_freq", type=int, default=10)
    parser.add_argument("--test_epoch_freq", type=int, default=10)
    parser.add_argument("--log_path", type=str, default="run.log")
    parser.add_argument("--ray_local_dir", type=str, default="~/ray_results")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--nesterov", action="store_true")
    # adamw
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--amsgrad", action="store_true")

    # scheduler
    # plateau
    parser.add_argument("--factor", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=10)
    # multistep
    parser.add_argument("--milestones", nargs='+', type=int, default=[80, 120])
    parser.add_argument("--gamma", type=float, default=0.1)
    # cosine
    parser.add_argument("--T_0", type=int, default=10)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument("--mult_gamma", type=float, default=0.999)
    parser.add_argument("--warmup_iter", type=int, default=10)

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
    model_choices = ("resnet18", "resnet50", "resnet101", "resnet152", "resnet_cifar")
    parser.add_argument("--model", type=str, default="resnet50", choices=model_choices)
    parser.add_argument("--classes", type=int, default=100)
    # distillation
    parser.add_argument("--teacher", type=str, default="resnet_cifar", choices=model_choices)
    parser.add_argument("--teacher_path", type=str, default="resnet_original_20.pth")
    parser.add_argument("--student", type=str, default="resnet18", choices=model_choices)
    parser.add_argument("--hard_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=20)

    # save model
    parser.add_argument("--save_model_cooldown", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="resnet50.pth")

    # data
    parser.add_argument("--data_root", type=str, default=r".")
    parser.add_argument("--extra_augment", action="store_true")
    # ColorJitter
    parser.add_argument("--ColorJitter", action="store_true")
    parser.add_argument("--brightness", type=float, default=0.5)
    parser.add_argument("--contrast", type=float, default=0.5)
    parser.add_argument("--saturation", type=float, default=0.5)
    parser.add_argument("--hue", type=float, default=0.5)
    # RandomAffine
    parser.add_argument("--RandomAffine", action="store_true")
    parser.add_argument("--degrees", type=float, default=90)
    parser.add_argument("--translate_M", type=float, default=0.5)
    parser.add_argument("--scale_m", type=float, default=0.5)
    parser.add_argument("--scale_M", type=float, default=2)
    parser.add_argument("--shear", type=float, default=45)
    # RandomPerspective
    parser.add_argument("--RandomPerspective", action="store_true")
    parser.add_argument("--distortion_scale", type=float, default=0.5)
    parser.add_argument("--perspective_p", type=float, default=0.5)
    # RandomGrayscale
    parser.add_argument("--RandomGrayscale", action="store_true")
    parser.add_argument("--gray_p", type=float, default=0.5)

    return parser


def get_cfg():
    parser = argparse.ArgumentParser()
    update_argument(parser)
    return parser.parse_args()


def cfg_to_str(cfg):
    cfg = vars(cfg)
    string = "\n".join([f"{k}: {v}" for k, v in cfg.items()])
    return string
