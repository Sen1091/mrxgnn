"""MR-XGNN: Modality-Robust eXplainable Graph Neural Network.

  1. Region-level node encoders for MRI and PET, plus shared ROI embeddings.
  2. Missing-modality handling: a learned mask token, or a cross-modal *hallucinator* that
     predicts PET node embeddings from the MRI graph. It is trained against real PET embeddings
     whenever PET exists.
  3. Region-level gated fusion: a per-ROI gate g_r in [0,1] gives how much each region relies
     on MRI versus PET. The gate is interpretable and is reported.
  4. Graph attention with an anatomical prior: attention logits get a bias of
     lambda * log(A_prior).
  5. Attention readout over nodes, then a clinical encoder (values + availability masks).
  6. Clinically anchored differentiable rules (NeuroSymAD-style rule functions, He et al. 2025),
     with three changes: a rule switches off when its input is missing; a penalty keeps the
     learned thresholds near guideline thresholds; each rule's contribution to each class logit
     is reported.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGraphAttention(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1, prior=None, prior_weight=1.0, use_prior=True):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.use_prior = use_prior and prior is not None
        if self.use_prior:
            self.register_buffer("logA", torch.log(prior.clamp_min(1e-4)))
            self.pw = nn.Parameter(torch.tensor(float(prior_weight)))
        self.last_attn = None

    def forward(self, x):
        B, R, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, R, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if self.use_prior:
            a = a + self.pw * self.logA
        a = a.softmax(-1)
        self.last_attn = a.detach()
        z = (self.drop(a) @ v).transpose(1, 2).reshape(B, R, d)
        x = x + self.drop(self.out(z))
        return x + self.drop(self.ff(self.n2(x)))


class RuleLayer(nn.Module):
    """delta_k(x) = softplus(a_k) * sigmoid((s*x - T1_k)/(tau_k*scale_k)) + softplus(b_k) * ReLU((s*x - T2_k)/scale_k)
    where s = +1 if higher values indicate disease and -1 otherwise. A rule is active only when x is
    observed. Class contribution = delta_k * w_k, where w_k in R^C is initialised from the rule's
    target class. T = T0 + scale * dT, and dT^2 is penalised so thresholds stay clinically anchored."""

    def __init__(self, rules, clin_names, rois, classes):
        n_classes = len(classes)
        super().__init__()
        self.rules, src, col, sign, T1, T2, sc, W0 = rules, [], [], [], [], [], [], []
        for r in rules:
            kind, var = r["var"].split(":", 1)
            src.append(0 if kind == "clin" else 1)
            col.append(clin_names.index(var) if kind == "clin" else rois.index(var))
            s = 1.0 if r.get("higher_is_disease", True) else -1.0
            sign.append(s)
            T1.append(s * r["t1"]); T2.append(s * r.get("t2", r["t1"])); sc.append(r.get("scale", 1.0))
            w = torch.zeros(n_classes)
            for cls, val in r.get("init_weights", {"AD": 1.0, "MCI": 0.5, "CN": -0.5}).items():
                if cls in classes:
                    w[classes.index(cls)] = val
            W0.append(w)
        K = len(rules)
        for name, v in (("src", src), ("col", col)):
            self.register_buffer(name, torch.tensor(v, dtype=torch.long))
        for name, v in (("sign", sign), ("T1_0", T1), ("T2_0", T2), ("scale", sc)):
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32))
        self.dT1 = nn.Parameter(torch.zeros(K)); self.dT2 = nn.Parameter(torch.zeros(K))
        self.a = nn.Parameter(torch.zeros(K)); self.b = nn.Parameter(torch.full((K,), -1.0))
        self.log_tau = nn.Parameter(torch.full((K,), math.log(0.5)))
        self.W = nn.Parameter(torch.stack(W0) * 0.5)

    def thresholds(self):
        return (self.sign * (self.T1_0 + self.scale * self.dT1)).detach(), \
               (self.sign * (self.T2_0 + self.scale * self.dT2)).detach()

    def forward(self, clin_raw, clin_mask, mri_w, m_mri, disabled=None):
        xs, ms = [], []
        for k in range(len(self.rules)):
            if self.src[k] == 0:
                xs.append(clin_raw[:, self.col[k]]); ms.append(clin_mask[:, self.col[k]])
            else:
                xs.append(mri_w[:, self.col[k]]); ms.append(m_mri)
        x = torch.stack(xs, 1) * self.sign
        m = torch.stack(ms, 1)
        T1 = self.T1_0 + self.scale * self.dT1
        T2 = self.T2_0 + self.scale * self.dT2
        tau = self.log_tau.exp() * self.scale
        delta = F.softplus(self.a) * torch.sigmoid((x - T1) / tau) + F.softplus(self.b) * F.relu((x - T2) / self.scale)
        delta = delta * m
        if disabled is not None:
            delta = delta * (1 - disabled)
        contrib = delta.unsqueeze(-1) * self.W          # [B, K, C]
        return contrib.sum(1), contrib

    def anchor_penalty(self):
        return (self.dT1 ** 2 + self.dT2 ** 2).mean()


class MRXGNN(nn.Module):
    def __init__(self, n_rois, n_clin, prior, classes=("CN", "MCI", "AD"), d=32, layers=2, heads=4, dropout=0.2,
                 use_mri=True, use_pet=True, use_clin=True, use_graph=True, use_prior=True,
                 use_halluc=True, fusion="gated", rules=None, clin_names=None, rois=None):
        super().__init__()
        classes = list(classes); n_classes = len(classes)
        self.cfg = dict(use_mri=use_mri, use_pet=use_pet, use_clin=use_clin, use_graph=use_graph,
                        use_halluc=use_halluc and use_pet and use_mri, fusion=fusion)
        self.enc_mri = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.enc_pet = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.roi_emb = nn.Parameter(torch.randn(n_rois, d) * 0.02)
        self.tok_mri = nn.Parameter(torch.zeros(d)); self.tok_pet = nn.Parameter(torch.zeros(d))
        if self.cfg["use_halluc"]:
            self.hall = nn.Sequential(DenseGraphAttention(d, heads, dropout, prior, use_prior=use_prior),
                                      nn.LayerNorm(d), nn.Linear(d, d))
        self.gate = nn.Linear(2 * d + 2, d)
        self.fuse_cat = nn.Linear(2 * d, d)
        self.gnn = nn.ModuleList([DenseGraphAttention(d, heads, dropout, prior, use_prior=use_prior)
                                  for _ in range(layers)]) if use_graph else \
            nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU()) for _ in range(layers)])
        self.pool = nn.Linear(d, 1)
        self.clin_enc = nn.Sequential(nn.Linear(2 * n_clin, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d)) if use_clin else None
        self.head = nn.Sequential(nn.LayerNorm(d * (2 if use_clin else 1)), nn.Dropout(dropout),
                                  nn.Linear(d * (2 if use_clin else 1), n_classes))
        self.rules = RuleLayer(rules, clin_names, rois, classes) if rules else None

    def node_embeddings(self, b):
        B = b["mri"].shape[0]
        m_m = b["m_mri"].view(B, 1, 1) * float(self.cfg["use_mri"])
        m_p = b["m_pet"].view(B, 1, 1) * float(self.cfg["use_pet"])
        h_m = self.enc_mri(b["mri"].unsqueeze(-1)) + self.roi_emb
        h_p = self.enc_pet(b["pet"].unsqueeze(-1)) + self.roi_emb
        h_m = m_m * h_m + (1 - m_m) * (self.tok_mri + self.roi_emb)
        h_p_real = h_p
        h_p_hat = None
        if self.cfg["use_halluc"]:
            h_p_hat = self.hall(h_m)
            h_p = m_p * h_p + (1 - m_p) * h_p_hat
        else:
            h_p = m_p * h_p + (1 - m_p) * (self.tok_pet + self.roi_emb)
        return h_m, h_p, h_p_real, h_p_hat, m_m, m_p

    def forward(self, b, disabled_rules=None):
        h_m, h_p, h_p_real, h_p_hat, m_m, m_p = self.node_embeddings(b)
        B, R, d = h_m.shape
        if not self.cfg["use_pet"]:
            h, g = h_m, torch.ones(B, R, 1)
        elif not self.cfg["use_mri"]:
            h, g = h_p, torch.zeros(B, R, 1)
        elif self.cfg["fusion"] == "gated":
            g = torch.sigmoid(self.gate(torch.cat([h_m, h_p, m_m.expand(B, R, 1), m_p.expand(B, R, 1)], -1)))
            h = g * h_m + (1 - g) * h_p
        else:
            h, g = self.fuse_cat(torch.cat([h_m, h_p], -1)), torch.full((B, R, 1), 0.5)
        for layer in self.gnn:
            h = layer(h) if isinstance(layer, DenseGraphAttention) else h + layer(h)
        w = self.pool(h).squeeze(-1).softmax(-1)             # [B, R] node importance (readout)
        z = (w.unsqueeze(-1) * h).sum(1)
        if self.clin_enc is not None:
            z = torch.cat([z, self.clin_enc(torch.cat([b["clin"] * b["clin_mask"], b["clin_mask"]], -1))], -1)
        logits = self.head(z)
        out = {"base_logits": logits, "gate": g.mean(-1), "readout": w,
               "h_p_real": h_p_real, "h_p_hat": h_p_hat}
        if self.rules is not None:
            add, contrib = self.rules(b["clin_raw"], b["clin_mask"], b["mri"], b["m_mri"], disabled_rules)
            logits = logits + add
            out["rule_contrib"] = contrib
        out["logits"] = logits
        return out
