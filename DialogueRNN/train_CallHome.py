import numpy as np
np.random.seed(1234)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from model import BiModel, Model, MaskedNLLLoss
from collections import OrderedDict


import pandas as pd

#from model import AVECModel, MaskedMSELoss

from dataloader import  CallHomeDataset


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = range(size)
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_CallHome_loaders(path, stance, batch_size=32, valid=None, num_workers=0, pin_memory=False, acproject=False, acfset="eGeMAPSv01a"):

    devset = CallHomeDataset(path=path, stance=stance, part="dev", acproject=acproject, acfset=acfset)
    testset = CallHomeDataset(path=path, stance=stance, part="eval", acproject=acproject, acfset=acfset)
    trainset = CallHomeDataset(path=path, stance=stance, part="train", acproject=acproject, acfset=acfset)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              #sampler=train_sampler,
                              shuffle=True,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(devset,
                              batch_size=batch_size,
                              #sampler=valid_sampler,
                              shuffle=True,
                              collate_fn=devset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    #testset = AVECDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def get_AVEC_loaders(path, batch_size=32, valid=None, num_workers=0, pin_memory=False):

    trainset = AVECDataset(path=path)

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = AVECDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    probs = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, metas  = [], [], [], []
    
    assert not train or optimizer!=None

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        
        if train:
            optimizer.zero_grad()

        # import ipdb;ipdb.set_trace()
        ## get batch features 
        textf, acouf, qmask, umask, label, metadf  =\
                [d.cuda() for d in data] if cuda else data
    #            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        ## qmask is the speakers
        ## umask is the sequence mask

        #log_prob = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask) # seq_len, batch, n_classes

        log_prob, alpha, alpha_f, alpha_b = model(textf, qmask, umask) # seq_len, batch, n_classes
        #print("log prob", log_prob.shape)
        lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes

        ## predictions are now batch first, but mask is seq first 
        #print("lp_", lp_.shape)

        labels_ = label.view(-1) # batch*seq_len

        currmask = umask.transpose(0,1).contiguous() #.view(-1)
        #loss = loss_function(lp_, labels_, umask)
        loss = loss_function(lp_, labels_, currmask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len

        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())

        currmask = currmask.view(-1)
        #print("currmask", currmask.shape) 
        #masks.append(umask.view(-1).cpu().numpy())
        masks.append(currmask.cpu().numpy())

        #print(masks) 
        #raise SystemExit
        probs.append(lp_.data.cpu().numpy())
        #print(metadf.shape)
        metas.append(metadf.view(-1))
        #print(metas[-1].shape)

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b

    if preds!=[]:
        preds  = np.concatenate(preds)
        probs  = np.concatenate(probs)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
        metas = np.concatenate(metas)
    #    print(probs.shape)
    #    print(metas.shape)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]


    ## Are these the wrong masks?
    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    avg_accuracy = round(accuracy_score(labels,preds,sample_weight=masks)*100,2)
    avg_fscore = round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)


    return avg_loss, avg_accuracy, labels, preds, probs, masks, avg_fscore, metas, [alphas, alphas_f, alphas_b]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.0,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs')

    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')

    parser.add_argument('--attention', default='simple', help='Attention type')

    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')

    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')

    parser.add_argument('--attribute', type=str , default="Positive", help='CallHome Stance')

    parser.add_argument('--encdir', default='/afs/inf.ed.ac.uk/user/c/clai/tunguska/stance2019/h5/encodings/', help='embedding directory')

    parser.add_argument('--acproject', action='store_true', default=False,
                        help='use acoustic encoding with projection layer')

    parser.add_argument('--acfset', type=str , default="eGeMAPSv01a", help='base acoustic feature set')

    args = parser.parse_args()


    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    n_classes  = 2 
    cuda       = args.cuda
    n_epochs   = args.epochs

    D_m = 100
    D_g = 100
    D_p = 100
    D_e = 100
    D_h = 100

    D_a = 100 # concat attention


    model = BiModel(D_m, D_g, D_p, D_e, D_h,
                    n_classes=n_classes,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    dropout_rec=args.rec_dropout,
                    dropout=args.dropout)

    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor([
                                        1., 
                                        1.
                                        ])
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)


    #print(args.attribute)

    ## basically we need a new data loader
    train_loader, valid_loader, test_loader =\
            get_CallHome_loaders(args.encdir, args.attribute, 
                                valid=0.0,
                                batch_size=batch_size,
                                num_workers=2, acproject=args.acproject, acfset=args.acfset)

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    ## This is doing early stopping on test, not dev?
    ## Also there's not model saving etc
    ## Check that it's using eval mode?

    patience = 10
    for e in range(n_epochs):
        start_time = time.time()
        if patience == 0:
            print("NO MORE PATIENCE", patience)
            break

        print("EPOCH:", e)

        train_loss, train_acc, _,_,_,_,train_fscore, _, _= train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)

        valid_loss, valid_acc, valid_label, valid_pred , valid_probs, valid_mask, val_fscore, val_metas, val_attentions = train_or_eval_model(model, loss_function, valid_loader, e)

        test_loss, test_acc, test_label, test_pred, test_probs, test_mask, test_fscore, test_metas, attentions = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_valid_loss > valid_loss:
            best_loss, best_label, best_pred, best_prob, best_mask, best_metas, best_attn =\
                    test_loss, test_label, test_pred, test_probs, test_mask, test_metas, attentions
            best_valid_loss, best_valid_label, best_valid_pred, best_valid_prob, best_valid_mask, best_valid_metas, best_valid_attn =\
                    valid_loss, valid_label, valid_pred, valid_probs, valid_mask, val_metas, val_attentions

            print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))
            patience = 10
        else:
                patience -= 1

        print("EPOCH/PATIENCE:", e, patience) 

        if args.tensorboard:
            writer.add_scalar('valid: accuracy/loss',valid_acc/valid_loss,e)
            writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
            writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)


    if args.tensorboard:
        writer.close()

    print('Dev performance..')
    print('Loss {} accuracy {}'.format(best_valid_loss,
                                     round(accuracy_score(best_valid_label,best_valid_pred,sample_weight=best_valid_mask)*100,2)))
    print(classification_report(best_valid_label,best_valid_pred,sample_weight=best_valid_mask,digits=4))
    print(confusion_matrix(best_valid_label,best_valid_pred,sample_weight=best_valid_mask))

    print('Test performance..')
    print('Loss {} accuracy {}'.format(best_loss,
                                     round(accuracy_score(best_label,best_pred,sample_weight=best_mask)*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


    print(np.exp(best_valid_prob))

    modname = "BiModel"
    Y_dev = best_valid_label
    devpred = best_valid_pred
    devprob = np.exp(best_valid_prob[:,1])
    print(Y_dev.shape, devpred.shape, devprob.shape)
    print(f1_score(Y_dev, devpred, sample_weight=best_valid_mask, pos_label=0))

    dev_metrics = pd.DataFrame(OrderedDict(
                [('classifier','DialogueRNN'), ('p1','turntrans_turnacoustic'), ('p2','X'), ('stance',args.attribute),
                ('f1',f1_score(Y_dev, devpred, sample_weight=best_valid_mask)),
                ('weighted_f1',f1_score(Y_dev, devpred, average='weighted', sample_weight=best_valid_mask)),
                ('precision',precision_score(Y_dev, devpred, sample_weight=best_valid_mask)),
                ('recall',recall_score(Y_dev, devpred, sample_weight=best_valid_mask)),

                ('f1_0',f1_score(Y_dev, devpred, sample_weight=best_valid_mask, pos_label=0)),
                ('weighted_f1_0',f1_score(Y_dev, devpred, average='weighted', sample_weight=best_valid_mask, pos_label=0)),
                ('precision_0',precision_score(Y_dev, devpred, sample_weight=best_valid_mask, pos_label=0)),
                ('recall_0',recall_score(Y_dev, devpred, sample_weight=best_valid_mask, pos_label=0)),

                ('accuracy',accuracy_score(Y_dev, devpred, sample_weight=best_valid_mask)),
                ('auroc',roc_auc_score(Y_dev, devprob, sample_weight=best_valid_mask))]
                        ), index=[modname])

    Y_eval = best_label
    evalpred = best_pred
    evalprob = np.exp(best_prob[:,1])
    eval_metrics = pd.DataFrame(OrderedDict(
                [('classifier','DialogueRNN'), ('p1','turntrans_turnacoustic'), ('p2','X'), ('stance',args.attribute),
                ('f1',f1_score(Y_eval, evalpred, sample_weight=best_mask)),
                ('weighted_f1',f1_score(Y_eval, evalpred, average='weighted', sample_weight=best_mask)),
                ('precision',precision_score(Y_eval, evalpred, sample_weight=best_mask)),
                ('recall',recall_score(Y_eval, evalpred, sample_weight=best_mask)),

                ('f1_0',f1_score(Y_eval, evalpred, sample_weight=best_mask, pos_label=0)),
                ('weighted_f1_0',f1_score(Y_eval, evalpred, average='weighted', sample_weight=best_mask, pos_label=0)),
                ('precision_0',precision_score(Y_eval, evalpred, sample_weight=best_mask, pos_label=0)),
                ('recall_0',recall_score(Y_eval, evalpred, sample_weight=best_mask, pos_label=0)),

                ('accuracy',accuracy_score(Y_eval, evalpred, sample_weight=best_mask)),
                ('auroc',roc_auc_score(Y_eval, evalprob, sample_weight=best_mask))]
                        ), index=[modname])

    alstr = "simple"
    if args.active_listener: 
        alstr = "active"

    acstr = "noproj"
    if args.acproject: 
        acstr = "acproj"
      
    acstr = acstr + "-" + args.acfset +  "-" + str(batch_size) 

    outdir = "./callhome"
    resfile =  outdir + "/DialogueRNN." + args.attribute +  "." + alstr + "." + acstr + ".metrics.txt"
    print(resfile)
    print(dev_metrics)
    dev_metrics.to_csv(resfile)

    resfile =  resfile + ".eval" 
    print(resfile)
    print(eval_metrics)
    eval_metrics.to_csv(resfile)


    predfile = "./preds/BiModel.turntrans_turnacoustic." + args.attribute + "." + alstr + "." + acstr + ".pred.dev.txt"
    print(predfile)
    preddf0 = pd.DataFrame({'classifier':'BiModel', 'p1':'X', 'xid':best_valid_metas, 'stance':args.attribute, 'pred':best_valid_pred, 'label':best_valid_label, 'mask':best_valid_mask})
    devprobdf = pd.DataFrame(np.exp(best_valid_prob), columns=["Pr0", "Pr1"]) 
    preddf = pd.concat([preddf0, devprobdf], axis=1)
    preddf.to_csv(predfile, index=False, sep="\t")

    predfile = predfile.replace(".dev.", ".eval.")  #"./preds/BiModel.turntrans_turnacoustic." + args.attribute + ".pred.eval.txt"
    print(predfile)
    preddf0 = pd.DataFrame({'classifier':'BiModel', 'p1':'X', 'xid':best_metas, 'stance':args.attribute, 'pred':best_pred, 'label':best_label, 'mask':best_mask})
    evalprobdf = pd.DataFrame(np.exp(best_prob), columns=["Pr0", "Pr1"]) 
    preddf = pd.concat([preddf0, evalprobdf], axis=1)
    preddf.to_csv(predfile, index=False, sep="\t")
