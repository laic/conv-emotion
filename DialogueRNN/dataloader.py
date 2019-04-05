import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import h5py 

class CallHomeDataset(Dataset):

    def __init__(self, path, stance, part="dev", acproject=False, acfset="eGeMAPSv01a"):
     
        print("PART", part, acproject)   
        filetemp = "mlp_[100]_STANCE_FEATNAME_eGeMAPSv01a_topseg_grustate_grustate_False_False_False_True_spksegid-segid_0.enc.h5"
        filetemp = filetemp.replace("STANCE", stance) 


        acfile = path + "/" + filetemp.replace("FEATNAME", "turnacoustic")
        if acproject:
            print("*** acproject")
            acfile = acfile.replace("False_False_False_True", "False_False_True_True")

        if acfset != "eGeMAPSv01a":
            print("*** acfset", acfset)
            acfile = acfile.replace("eGeMAPSv01a", acfset)


        print(acfile)
        hfac = h5py.File(acfile, "r")
        partacdf, acnames = self.get_h5_part(hfac, part, featname="turnacoustic")       
        hfac.close()
 
        print(partacdf.head())
        print(partacdf.shape)

        txtfile = path + "/" + filetemp.replace("FEATNAME", "turntrans")
        print(txtfile)
        hftxt = h5py.File(txtfile, "r")
        parttxtdf, txtnames = self.get_h5_part(hftxt, part, featname="turntrans")       
        hftxt.close()

        print(parttxtdf.head())
        print(parttxtdf.shape)

        partdf = parttxtdf.merge(partacdf, how='inner')
        partdf['target'] = partdf['target'].astype(int)
        partdf.loc[:,'xid'] = range(partdf.shape[0]) 

        metacols = ["xid", "spksegid", "segid", "conv", "fname", "spk", "topic", "target"]
        hfmeta = partdf[metacols]       
        hfmeta.to_csv("./meta/" + stance + "." + part + ".meta.txt")
 
        ## One dictionary per clip = fname 
        self.part_clips = partdf.fname.unique()
        self.part_acoustic = {} 
        self.part_text = {} 
        self.part_speakers = {}
        self.part_labels = {}
        self.part_meta = {}

        #self.keys = self.part_clips[:100]
        self.keys = self.part_clips 

        for clip in self.keys:
            #print(clip) 
            currac = partdf[partdf.fname == clip][acnames]
            self.part_acoustic[clip] = list(np.array(currac))

            currtxt = partdf[partdf.fname == clip][txtnames]
            self.part_text[clip] = list(np.array(currtxt))

            currspk = partdf[partdf.fname == clip]['spk']
            self.part_speakers[clip] = list(currspk)

            currlabels = partdf[partdf.fname == clip]['target']
            self.part_labels[clip] = list(currlabels)

            currmeta = partdf[partdf.fname == clip][['xid']].astype(int)
            self.part_meta[clip] = list(np.array(currmeta))

        print(len(self.part_acoustic))
        print(len(self.part_text))
        print(len(self.part_speakers))
        print(len(self.part_labels))
        print(len(self.part_clips))
        print(len(self.part_meta))

        self.len = len(self.keys)

        print("FINISHED INIT", part)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.part_text[vid]),\
               torch.FloatTensor(self.part_acoustic[vid]),\
               torch.FloatTensor([[1,0] if x=='A' else [0,1] for x in\
                                  self.part_speakers[vid]]),\
               torch.FloatTensor([1]*len(self.part_labels[vid])),\
               torch.LongTensor(self.part_labels[vid]) , \
               torch.LongTensor(self.part_meta[vid])

    def get_h5_part(self, hfac, part, featname='turnacoustic'):

        part_acoustic = pd.DataFrame(np.array(hfac[part][featname]))
        fnames = [ featname + str(i) for i in range(1,part_acoustic.shape[1]+1)]
        part_acoustic = part_acoustic.rename(columns=dict(zip(part_acoustic.columns.values, fnames)))

        metacols = ['spksegid','spksegid.1','segid','segid.1','conv','fname','spk','topic','stancename','target']
        part_meta = pd.DataFrame(np.array(hfac[part]['meta']).astype(str), columns=metacols)
        part_meta = part_meta.drop(['spksegid.1', 'segid.1'], axis=1)

        partdf = pd.concat([part_meta, part_acoustic], axis=1)

        starttimes = partdf.segid.apply(lambda u: int(u.split("_")[2].split("-")[0]))
        partdf.loc[:,'starttime'] = starttimes

        ## now we need to reorder based on starttimes
        partdf = partdf.sort_values(by=['conv','fname','starttime'])
        print(type(partdf))
        print(type(fnames))
        return (partdf, fnames)


    def __len__(self):
        return self.len

    def collate_fn(self, data):
        #print("collate_fn")
        dat = pd.DataFrame(data)
        #print(dat.head())
        #print(dat.shape)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]


class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        '''
        curr = list(self.videoSpeakers)[0]
        print(curr)
        print(self.videoSpeakers[curr])

        print(self.videoSpeakers[curr])
        print(type(self.videoSpeakers[curr]))
        print(self.videoLabels[curr])
        print(type(self.videoLabels[curr]))
        print(type(self.videoAudio[curr]))
        print(type(self.videoAudio[curr][0]))
        print(type(self.videoText[curr]))
        print(type(self.videoText[curr][0]))

        raise SystemExit
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]

class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence,\
            self.trainVid, self.testVid = pickle.load(open(path, 'rb'),encoding='latin1')


        #print(self.videoAudio[2][0])
        print(self.videoAudio[2][0].shape)
        print(len(self.videoAudio))
        #for k, v in self.videoAudio.items():
        #        print(k, len(v))


        print (len(self.trainVid))
        print (len(self.testVid))

        print(type(self.videoSpeakers))

        print(self.videoSpeakers[2])
        print(type(self.videoSpeakers[2]))
        print(self.videoLabels[2])
        print(type(self.videoLabels[2]))
        print(type(self.videoAudio[2]))
        print(type(self.videoAudio[2][0]))
        print(type(self.videoText[2]))
        print(type(self.videoText[2][0]))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='user' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        #print("collate_fn")
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) for i in dat]

class MELDDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoSpeakers[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<5 else dat[i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):
        
        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))
        
        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]
        
        return  torch.LongTensor(self.InputSequence[conv]), \
                torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.Speakers[conv]]),\
                torch.FloatTensor([1]*len(self.ActLabels[conv])), \
                torch.LongTensor(self.ActLabels[conv]), \
                torch.LongTensor(self.EmotionLabels[conv]), \
                self.InputMaxSequenceLength[conv], \
                conv

    def __len__(self):
        return self.len
    


class DailyDialoguePadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):

        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))
        
        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]
        
        # stack all
        return torch.stack(batch, dim=0)
    
    def __call__(self, batch):
        dat = pd.DataFrame(batch)
        
        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i==0 else \
                pad_sequence(dat[i]) if i == 1 else \
                pad_sequence(dat[i], True) if i < 5 else \
                dat[i].tolist() for i in dat]
