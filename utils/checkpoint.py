import os
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import json
import datetime
import pickle

def saveckpt(path='checkpoint', id_=None, **kwargs):
    '''saves checkpoint from a given set of kwargs
    path is the only mandatory input'''
    db = {
        # 'time':str(datetime.datetime.now())[:-7],
        'title':None,
        'note':None,
        'time':str(datetime.datetime.now()),
        'data':{}
    }

    if id_==None:
        id_ = kwargs[list(kwargs.keys())[0]]
    # print('identifier:', id_)

    if not os.path.isdir(path):
        os.mkdir(path)
    
    for i in kwargs.items():
        key = i[0]
        item = i[1]
        if isinstance(item, (int, float, str)):
            db[key] = item

        else:
            patht = 'ckpt-%s_%i.pth'%(key,id_)
            db[key] = {
                'type' :type(item).__name__,
                'path' : patht,
            }
            patht = os.path.join(path, patht)
            torch.save(item.state_dict(), patht)


    pathj = os.path.join(path, 'ckpt-%s_%i.json'%('db',id_))
    with open(pathj, 'w', encoding='utf-8') as f:
        f.write(json.dumps(db, ensure_ascii=False, indent=4))
        print('\x1b[32mckpt saved to %s\x1b[39m'%pathj)

def loadckp(fetch:(list, tuple, str)=None, path=None, id_=None, filename=None, device=None, **kwargs,):
    '''
    filename = path + filename
    '''
    # assert filename is not None
    # filename = os.path.join(path, filename)
    if filename is not None:
        pass
    elif id_ is not None:
        filename = os.path.join(path, 'ckpt-db_%i.json'%id_)
    else:raise Exception
    with open(filename) as f:
        db = json.loads(f.read())

    dirname = os.path.dirname(filename)

    if device==None: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if isinstance(fetch, str):
        fetch = [fetch,]
    elif fetch is None: fetch = []
    
    for i in kwargs.keys():
        fetch.append(i)

    out = []
    for key in fetch:
        item = db[key]
        if isinstance(item, (int, float, str)):
            out.append(item)
        elif isinstance(item, dict):
            file = item['path']
            # file = torch.load(os.path.join(dirname, file))
            file = torch.load(os.path.join(dirname, file), map_location=device)
            kwargs[key].load_state_dict(file)

    print('ckpt loaded (saved@ %s):\n    '%db['time'][:-7], fetch, '\n    ', out)

    return out


if __name__=="__main__":
    import torchvision as tv
    net = tv.models.resnet18()
    saveckpt2(epoch=1, loss=0.02109, net=net, title='test')
    print(loadckpt2('epoch', filename='checkpoint/ckpt-db-1.json', net=net))
