"""Minimal M1: region-conditioned state writing.

This is a mechanism prototype, not a claimed final method. The lower and
upper hidden partitions receive separate write gates; no labels or future
frames enter the gate.
"""
import torch
from torch import nn
from .baseline import rotation6d


class RegionEventWrite(nn.Module):
    def __init__(self,hidden=192):
        super().__init__()
        if hidden%3: raise ValueError('hidden must be divisible by 3')
        self.hidden=hidden;self.enc=nn.Linear(66,hidden);self.cell=nn.GRUCell(hidden,hidden)
        # 5-sensor blocks are 13 values each; pelvis+shanks vs forearms.
        self.lower_gate=nn.Sequential(nn.Linear(39,hidden//3),nn.Sigmoid())
        self.upper_gate=nn.Sequential(nn.Linear(26,hidden//3),nn.Sigmoid())
        self.head=nn.Linear(hidden,23*6)
        nn.init.normal_(self.head.weight,std=1e-3)
        with torch.no_grad():self.head.bias.copy_(torch.tensor([1.,0,0,0,1,0]).repeat(23))

    def forward(self,x,state=None,return_gates=False):
        if x.ndim!=3 or x.shape[-1]!=66:raise ValueError('Expected [batch,time,66]')
        b,t,_=x.shape
        if state is None:state=x.new_zeros((b,self.hidden))
        elif state.shape!=(b,self.hidden):raise ValueError('Invalid recurrent state')
        lower=[];upper=[];outs=[]
        for i in range(t):
            candidate=self.cell(torch.tanh(self.enc(x[:,i])),state)
            gl=self.lower_gate(x[:,i,:39]);gu=self.upper_gate(x[:,i,39:65])
            k=self.hidden//3;old=state.view(b,3,k);new=candidate.view(b,3,k)
            mixed=torch.cat(((1-gl)*old[:,0]+gl*new[:,0],(1-gu)*old[:,1]+gu*new[:,1],
                             (1-(gl+gu)/2)*old[:,2]+((gl+gu)/2)*new[:,2]),dim=-1)
            state=mixed;outs.append(state);lower.append(gl);upper.append(gu)
        h=torch.stack(outs,1);local=rotation6d(self.head(h).view(b,t,23,6));root=x[...,:9].view(b,t,1,3,3)
        pred=torch.cat((root,local),dim=-3)
        if return_gates:return pred,state,torch.stack(lower,1),torch.stack(upper,1)
        return pred,state
