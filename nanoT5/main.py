import time

import hydra
import torch
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


@hydra.main(config_path="configs", config_name="default", version_base="1.1")
def main(args):
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16) # thank god for jank

    logger = setup_basics(args)
    tokenizer = get_tokenizer(args)

    config = get_config(args, tokenizer)
    model = get_model(args, config)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)
    model_summary(model, max_depth=3)
    model.config.save_pretrained(".")

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
