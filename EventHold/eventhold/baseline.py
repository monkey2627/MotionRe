"""Small causal baseline; no event-memory claim and no test-label access."""
import torch
from torch import nn
from torch.nn import functional as F


def rotation6d(x):
    a,b=x[...,:3],x[...,3:]
    u=F.normalize(a,dim=-1,eps=1e-8)
    v=F.normalize(b-(u*b).sum(-1,keepdim=True)*u,dim=-1,eps=1e-8)
    return torch.stack((u,v,torch.cross(u,v,dim=-1)),dim=-1)


def geodesic(pred,target):
    delta=pred.transpose(-1,-2)@target
    # atan2 is accurate at zero; clamped acos alone produces a nonzero floor.
    skew=torch.stack((delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],
                      delta[...,1,0]-delta[...,0,1]),dim=-1)/2
    sin=torch.linalg.vector_norm(skew,dim=-1)
    cos=((delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    return torch.atan2(sin,cos)


class CausalBaseline(nn.Module):
    def __init__(self,hidden=256,layers=2):
        super().__init__()
        self.gru=nn.GRU(66,hidden,num_layers=layers,batch_first=True)
        self.head=nn.Linear(hidden,23*6)
        nn.init.normal_(self.head.weight,std=1e-3)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.,0,0,0,1,0]).repeat(23))

    def forward(self,x,state=None):
        if x.ndim!=3 or x.shape[-1]!=66:
            raise ValueError('Expected EventHold [batch,time,66] features')
        h,state=self.gru(x,state)
        local=rotation6d(self.head(h).reshape(*h.shape[:2],23,6))
        # The measured bone-calibrated pelvis rotation is the only root input.
        root=x[...,:9].reshape(*x.shape[:2],1,3,3)
        return torch.cat([root,local],dim=-3),state
