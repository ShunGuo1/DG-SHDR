from option import args
import torch
import loss
import model
from trainer1 import Trainer
import dataloader


if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(42)
    model = model.Model(args)
    loader = dataloader.loader(args)
    loss = loss.Loss(args)
    t = Trainer(args, loader, model, loss)
    epoch = 0
    if args.load =="1":
        t.test1_result(epoch)
    else:
        #t.inference(epoch)
        t.test2_result1(epoch)
        #t.test3_result(epoch)
