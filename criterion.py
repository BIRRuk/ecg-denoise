import torch
import torchvision as tv

class CustomLoss(torch.nn.Module):
    def __init__(self, class_weight=None):
        super().__init__()
        # if class_weight is not None:
        #     # self.alpha *= class_weight
        #     # self.alpha_neg *= class_weight

        self.bce_logits = torch.nn.BCEWithLogitsLoss(weight=class_weight)
        self.mse = torch.nn.MSELoss()
        print(f'[ {__name__} ] custom loss initialized')
    
    def forward(self, x, y):
        rx = x[:,:-1]
        ry = y[:,:-1]
        # print(ry)
        # r_loss = self.bce_logits(rx, ry)
        # r_loss = tv.ops.sigmoid_focal_loss(rx, ry.type_as(rx), reduction='mean')
        r_loss = tv.ops.sigmoid_focal_loss(rx, ry, reduction='mean')

        # return r_loss

        signal_x = x[:,-1]
        signal_y = y[:,-1]
        signal_loss = self.mse(signal_x, signal_y.type_as(signal_x))

        return (.5*r_loss)+(.5*signal_loss)


class CustomMetric():
    def __init__(self, classes_out, batch_size=None, device=None):
        super().__init__()
        # self.correct = torch.zeros(classes_out, device=device)
        self.total_el = 0
        self.correct = 0
        self.total_count = 0
        self.bs = batch_size
        self.classes_out = classes_out

    def forward(self, x, y):
        # bs,_, len_ = x.shape[0], x.shape[-1]
        bs,_, len_ = x.shape


        rx = x[:,:-1]
        ry = y[:,:-1]
        rnumel = ry.numel()

        signal_x = x[:,-1]
        signal_y = y[:,-1]

        rx = torch.relu(torch.sign(rx))
        rx = (rx==ry).int().sum() # !change dtype from int32
        # rx = rx.sum(dim=-1)/rx.shape[-1]

        self.total_count += bs
        self.total_el += rnumel
        self.correct += rx

        return rx.sum().item()/rnumel, bs

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def accuracy(self, mode='primary'):
        if mode=='primary' : return self.correct.sum()/self.total_el
        else: return {'acc_val': self.correct/self.total_el}

    def stat(self, detailed=False):
        print('accuracy: {}'.format(self.accuracy(separate=detailed).tolist()))


metrics = { 'metric':CustomMetric, 'val_metric':CustomMetric,}

if __name__ == '__main__':
    metric = CustomMetric(2)
    print(metric)
