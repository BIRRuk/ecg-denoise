import torch


class BCEAcc():
    def __init__(self, classes_out, batch_size=None, device=None):
        super().__init__()
        self.correct = torch.zeros(classes_out, device=device)
        self.total_count = 0
        self.bs = batch_size
        self.classes_out = classes_out

    def forward(self, x, y):
        bs = x.shape[0]
        x = torch.relu(torch.sign(x))
        x = (x==y).int().sum(dim=0) # !change dtype from int32

        self.total_count += bs
        self.correct += x

        return x.sum().item()/bs/self.classes_out, bs

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def accuracy(self, mode='primary'):
        if mode=='primary' : return self.correct.sum()/(self.total_count*self.classes_out)
        else: return {'acc_val': self.correct/self.total_count}

    def stat(self, detailed=False):
        print('accuracy: {}'.format(self.accuracy(separate=detailed).tolist()))


class BCEMed():
    '''BCEMed calculates count of true posetives, true negatives, total posetives, and total negatives 
    addition to accuracy calculated in BCEAccS'''
    def __init__(self, classes_out, batch_size=None, device=None):
        super().__init__()
        self.correct = torch.zeros(classes_out, device=device)
        self.tp = torch.zeros(classes_out, device=device)#, dtype=torch.int)
        self.tn = torch.zeros(classes_out, device=device)
        self.pos = torch.zeros(classes_out, device=device)
        self.neg = torch.zeros(classes_out, device=device)
        
        self.total_count = 0
        self.bs = batch_size
        self.classes_out = classes_out

    def to(self, device):
        for i in (self.correct, self.tp, self.tn, self.pos, self.neg): i.to(device)

    def forward(self, x, y):
        bs = x.shape[0]
        x = torch.relu(torch.sign(x))
        x = (x==y).int() # !change dtype from int32

        # sensetivity
        tp = (x*y).sum(dim=0)
        self.tp += tp
        pos = y.sum(dim=0)
        self.pos += pos

        # specificity
        neg = -y+1
        tn = (x*neg).sum(dim=0)
        self.tn += tn
        self.neg += neg.sum(dim=0)

        # overall accuracy
        acc = x.sum(dim=0)
        self.correct += acc

        self.total_count += bs
        return acc.sum().item()/bs/self.classes_out, bs

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def accuracy(self, mode='primary'):
        if mode=='primary':
            return self.correct.sum()/(self.total_count*self.classes_out)
        elif mode=='trn':
            return {'acc_trn': self.correct.sum()/(self.total_count*self.classes_out)}
        elif mode=='val':
            return self.acc_med()
        else: raise Exception

    def acc_med(self):
        overall = {
            'acc_val' : self.correct.sum()/(self.total_count*self.classes_out),
            'sensetivity': self.tp.sum()/self.pos.sum(),
            'specificity': self.tn.sum()/self.neg.sum(),
            'dice' : (self.tp*self.tn).sum()/((self.pos-self.tp)*(self.neg-self.tn)).sum(),

            'accs' : self.correct/self.total_count,
            'senss': self.tp/self.pos,
            'specs': self.tn/self.neg,
            'dices' : (self.tp*self.tn)/((self.pos-self.tp)*(self.neg-self.tn)),
        }
        return overall
        
    def print_stats(self, mode='val',):
        data = self.accuracy(mode=mode)
        for i in data.items():
            print(i)

    def db(self):
        return {
            'corrects':self.correct,#
            'true_pos':self.tp,#
            'true_neg':self.tn,
            'total_pos':self.pos,#
            'total_neg':self.neg,#.tolist()
            'total_count':self.total_count,
            'classes_out':self.classes_out,
        }


class SoftmaxAcc():
    """docstring for Softmax_Acc"""
    def __init__(self, classes_out, batch_size=None):
        super().__init__()
        self.correct = torch.zeros(classes_out)
        self.tp = torch.zeros(classes_out)#, dtype=torch.int)
        self.tn = torch.zeros(classes_out)
        self.pos = torch.zeros(classes_out)
        self.neg = torch.zeros(classes_out)
        
        self.total_count = 0
        self.bs = batch_size
        self.classes_out = classes_out

    # def __repr__(self): 
    #     return "__repr__"
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, x, labels) -> float:
        bs = labels.shape[0]
        self.total_count += bs

        _, preds = torch.max(x, dim=1)

        x = self.one_hot(preds)
        y = self.one_hot(labels)

        bs = x.shape[0]
        x = torch.relu(torch.sign(x))
        x = (x==y).int() # !change dtype from int32

        # sensetivity
        tp = (x*y).sum(dim=0)
        self.tp += tp
        pos = y.sum(dim=0)
        self.pos += pos

        # specificity
        neg = -y+1
        tn = (x*neg).sum(dim=0)
        self.tn += tn
        self.neg += neg.sum(dim=0)

        # overall accuracy
        acc = x.sum(dim=0)
        self.correct += acc

        self.total_count += bs
        return tp.sum().item()/pos.sum().item()


    def one_hot(self, x):
        bs = x.shape[0]
        out = torch.zeros(bs, self.classes_out)
        out[torch.arange(bs), x] = 1.
        return out

    def acc_med(self):
        overall = {
            # 'acc_val' : self.correct.sum()/(self.total_count*self.classes_out),
            'acc_val' : self.tp.sum()/self.pos.sum(),
            'sensetivity': self.tp.sum()/self.pos.sum(),
            'specificity': self.tn.sum()/self.neg.sum(),
            'dice' : (self.tp*self.tn).sum()/((self.pos-self.tp)*(self.neg-self.tn)).sum(),

            'accs ' : self.correct/self.total_count,
            'senss': self.tp/self.pos,
            'specs': self.tn/self.neg,
            'dices' : (self.tp*self.tn)/((self.pos-self.tp)*(self.neg-self.tn)),
        }
        return overall
        
    def accuracy(self, mode='primary'):
        if mode=='primary':
            # return self.correct.sum()/(self.total_count*self.classes_out)
            return self.tp.sum()/self.pos.sum()
        elif mode=='trn':
            # return {'acc_trn': self.correct.sum()/(self.total_count*self.classes_out)}
            return {'acc_trn': self.tp.sum()/self.pos.sum()}
        elif mode=='val':
            return self.acc_med()
        else: raise Exception

    def print_stats(self, mode='val',):
        data = self.accuracy(mode=mode)
        for i in data.items():
            print(i)
    # def print_stats(self, mode='val',):
    #     import json
    #     print(json.dumps(self.accuracy(mode=mode), indent=4))

class SoftmaxCorrect():
    """docstring for Softmax_Acc"""
    def __init__(self, classes_out, batch_size=None):
        super().__init__()
        self.correct = 0
        
        self.total_count = 0
        self.classes_out = classes_out

    # def __repr__(self): 
    #     return "__repr__"
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, x, labels) -> float:
        bs = labels.shape[0]
        self.total_count += bs
        _, preds = torch.max(x, dim=1)
        preds = (preds==labels).sum()
        acc = preds/bs
        self.correct += acc
        return acc
        
    def accuracy(self, mode='primary'):
        return self.correct/self.total_count
        if mode=='primary':
            return self.correct/self.total_count
        elif mode=='trn':
            return {'acc_trn': self.correct/self.total_count}
        elif mode=='val':
            return {'acc_val': self.correct/self.total_count}
        else: raise Exception

    def print_stats(self, detailed=False):
        print('accuracy: {}'.format(self.accuracy()))


metrics = {
    # "bce_acc":bce_acc,
    # "softmax_correct":softmax_correct,
    # 'sensetivity':sensetivity,
    # 'specificity':specificty,
    # 'dice_score':dice_score,
    "bce_acc_cls":BCEAcc,
    'bce_med':BCEMed,
    'softmax_acc_med_cls':SoftmaxAcc,
    'softmax_correct_cls':SoftmaxAcc,
}

m_config = {
    'BCEWithLogitsLoss':{ 'metric':BCEAcc, 'val_metric':BCEMed,},
    'Softmax':{'metric':SoftmaxAcc, 'val_metric':SoftmaxAcc},
}

if __name__=='__main__':
    EXAMPLES = 7
    CLASSES = 4
    metric = metrics['softmax_acc_cls'](CLASSES)
    print(metric)
    print(metric(torch.randn(EXAMPLES,CLASSES), torch.randint(0,CLASSES,(EXAMPLES, ),)))
    print(metric.accuracy())
    print(metric.accuracy(mode='val'))
