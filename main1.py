from option import args
import torch
import numpy as np
import random
import loss
import model
from trainer1 import Trainer
import dataloader


if __name__ == '__main__':
    # 早停机制训练
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    epoch = 1
    if args.train:
        model = model.Model(args)
        #loader = dataloader.VimeoLoader(args)
        loader = dataloader.loader(args)
        loss = loss.Loss(args)
        t = Trainer(args, loader, model, loss)
        while epoch < 1000:
            epoch, stage = t.train(epoch)
            if stage == 1:
                stage = t.test1_nocut(epoch)
                #stage = t.tt(epoch)
            elif stage == 2:
                stage = t.test2_nocut(epoch)
            if stage == -1:
                print("model finished")
                break
            epoch += 1