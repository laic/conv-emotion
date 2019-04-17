import pandas as pd
import sys

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from collections import OrderedDict


def get_pred_dist(udf): 
        n0 = udf[udf.pred==0].shape[0]
        n1 = udf[udf.pred==1].shape[0]
        nturns = udf.shape[0]
        pred = (n1 >  n0) * 1 
        predmix = ((n1 > 0) and (n0 > 0))
        xidfirst  = udf['xid'].iloc[0]
        xidlast  = udf['xid'].iloc[-1]
        prob = udf['Pr1'].mean()

        target = udf['target'].unique() 
        label = udf['label'].unique() 
        if target.shape[0] != 1:
                print "more than one target for seg?", target
                raise SystemExit
        currname = udf['spksegid'].unique()

        return pd.DataFrame({'xidf':xidfirst, 'xidl':xidlast, 'n0':n0, 'n1':n1, 'nturns':nturns, 'pred':pred, 'prob':prob, 'target':target, 'label':label, 'predmix':predmix}, index=[currname]) 


print sys.argv[1]
#stance = "Positive"
#part = "dev"
stance = sys.argv[1]
part = sys.argv[2]
listener = sys.argv[3]

preddir ='/afs/inf.ed.ac.uk/user/c/clai/tunguska/emo/conv-emotion/DialogueRNN/preds/'
#./preds/BiModel.turntrans_turnacoustic.Interested.simple.acproj.pred.dev.txt
predfile = preddir + "/BiModel.turntrans_turnacoustic." + stance + "." + listener + ".acproj-IS13_ComParE.pred." + part + ".txt"
print predfile


xdf = pd.read_csv(predfile, sep="\t")
#xdf = xdf.rename(columns={'val':'pred', 'pred':'label'})
xdf.head()


ind0 = xdf.index[xdf.xid == 1][0]-1
#print xdf.loc[ind0]
xdf.loc[ind0, "xid"] = -1
#print xdf[xdf.xid == -1]
xdf = xdf[xdf.xid != 0].copy()
xdf.loc[xdf.xid == -1, 'xid'] = 0
print xdf.shape
#print xdf[xdf.xid==0]

metadir = '/afs/inf.ed.ac.uk/user/c/clai/tunguska/emo/conv-emotion/DialogueRNN/meta/'
metafile = metadir + "/" + stance + "." + part + ".meta.txt" 
metadf = pd.read_csv(metafile)
print metadf.head()
print metadf.shape

resdf = metadf.merge(xdf, on='xid', how='inner')
print(resdf.head())
print(resdf.shape)


#ressum = resdf.iloc[0:200].groupby(['spksegid','fname','spk']).apply(get_pred_dist)
ressum = resdf.groupby(['spksegid','fname','spk','topic']).apply(get_pred_dist)
#print(ressum.reset_index()) #.drop('level_1', axis=1))
resfile = predfile + "ressum.txt"
ressum.to_csv(resfile)
print ressum.head()

Y_dev = ressum['target']  
devpred = ressum['pred'] 
devprob = ressum['prob'] 

print confusion_matrix(Y_dev, devpred)

part_metrics = pd.DataFrame(OrderedDict(
                [('classifier','BiModel'), ('p1','X'), ('p2','majority'), ('stance',stance),

                ('f1',f1_score(Y_dev, devpred)),
                ('precision',precision_score(Y_dev, devpred)),
                ('recall',recall_score(Y_dev, devpred)),

                ('f1_0',f1_score(Y_dev, devpred, pos_label=0)),
                ('precision_0',precision_score(Y_dev, devpred, pos_label=0)),
                ('recall_0',recall_score(Y_dev, devpred, pos_label=0)),

                ('accuracy',accuracy_score(Y_dev, devpred)),
                ('weighted_f1',f1_score(Y_dev, devpred, average='weighted')), 
                ('auroc',roc_auc_score(Y_dev, devprob))]
                        ), index=[0])

print "ANY MIX?"
print ressum[ressum.predmix == True]

print part_metrics
outfile = predfile + ".maj.metrics.txt"
print outfile
part_metrics.to_csv(outfile)

