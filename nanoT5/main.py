import time

import hydra
import torch
from accelerate import Accelerator
from omegaconf import open_dict

from .utils import (
    eval,
    get_config,
    get_dataloaders,
    get_lr_scheduler,
    get_model,
    get_optimizer,
    get_tokenizer,
    model_summary,
    predict,
    setup_basics,
    train,
)

import cProfile
import pstats

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(func, *args, **kwargs)
        finally:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative').print_stats(20)
    return wrapper

# @profile_function
@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(args):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
    )
    logger = setup_basics(accelerator, args)
    tokenizer = get_tokenizer(args)

    config = get_config(args, tokenizer)
    model = get_model(args, config)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    # Resize token embeddings if necessary
    model.resize_token_embeddings(len(tokenizer))

    logger.log_args(args)
    model_summary(model, max_depth=3)
    model.config.save_pretrained(".")

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    if args.model.liger:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    if args.model.compile:
        model = torch.compile(model)

    

    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, test_dataloader, logger, args, tokenizer)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger, args, tokenizer)
    else:

        train(
            model,
            train_dataloader,
            test_dataloader,
            accelerator,
            lr_scheduler,
            optimizer,
            logger,
            args,
            tokenizer,
        )

    tokenizer.save_pretrained("./tokenizer")
    logger.finish()


if __name__ == "__main__":
    main()
