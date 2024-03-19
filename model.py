"""Implementation of the Specformer Model"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SineEncoding(nn.Module):
    """Eigen Encoding"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        ee = e * self.constant
        div = torch.exp(
            torch.arange(0, self.hidden_dim, 2) * (-math.log(10000) / self.hidden_dim)
        )
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class SpecLayer(nn.Module):
    """Spectral Layer"""
    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm="none"):
        super().__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == "none":
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == "layer":
            self.norm = nn.LayerNorm(ncombines)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(ncombines)
        else:
            self.norm = None

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class Specformer(nn.Module):
    def __init__(
        self,
        nclass,
        nfeat,
        nlayer=1,
        hidden_dim=128,
        nheads=1,
        tran_dropout=0.0,
        feat_dropout=0.0,
        prop_dropout=0.0,
        norm="none",
    ):
        super().__init__()
        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_form = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        if norm == "none":
            self.layers = nn.ModuleList(
                [
                    SpecLayer(nheads + 1, nclass, prop_dropout, norm=norm)
                    for i in range(nlayer)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    SpecLayer(nheads + 1, hidden_dim, prop_dropout, norm=norm)
                    for i in range(nlayer)
                ]
            )

    def forward(self, e, u, x):
        e.size(0)
        ut = u.permute(1, 0)

        if self.norm == "none":
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        eig = self.eig_encoder(e)

        mha_eig = self.mha_form(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig)

        for conv in self.layers:
            basic_feats = [h]
            utx = ut @ h
            for i in range(self.nheads):
                basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))
            basic_feats = torch.stack(basic_feats, dim=1)
            h = conv(basic_feats)

        if self.norm == "none":
            return h, new_e
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h, new_e
