"""
Model definitions and training for split detection.

- SplitDetector: MLP encoder + multi-class classifier, trained with CE + triplet loss
- ProjectionHead: legacy linear projection (kept for checkpoint compatibility)
- train_detector: joint CE + triplet training with balanced sampling
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ALL_LABELS, LABEL2ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class ProjectionHead(nn.Module):
    """Single linear projection with L2-normalization output."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.fc.weight)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)

    def embed(self, x):
        return self.forward(x)


class SplitDetector(nn.Module):
    """MLP encoder + multi-class classifier for split detection.

    Joint training with CE (multi-class) + triplet (split-focused) loss.
    At inference, use P(split) from the classifier for scoring, and the
    L2-normalised embedding for gallery-based visualisation.
    """
    def __init__(self, in_dim: int, embed_dim: int = 256,
                 n_classes: int = len(ALL_LABELS)):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        """L2-normalised embedding (for triplet loss & retrieval)."""
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x):
        """Returns (embedding_normed, logits)."""
        e = self.encoder(x)
        e_norm = F.normalize(e, dim=-1)
        logits = self.classifier(e)
        return e_norm, logits

    def predict_proba(self, x):
        """Class probabilities (N, n_classes)."""
        with torch.no_grad():
            _, logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# Triplet dataset with online hard-negative mining
# --------------------------------------------------------------------------- #

class TripletSampler:
    """
    Pre-indexes positive / negative sets and yields numpy batches.
    Supports class-weighted negative sampling to up-weight hard confusers.
    Avoids torch.from_numpy / DataLoader to sidestep system-torch NumPy ABI issues.
    """
    def __init__(self, items, focus_label: str = "split", neg_oversample: int = 8,
                 hard_neg_labels: list = None, hard_neg_weight: float = 5.0):
        """
        hard_neg_labels : guaranteed in every batch's negative pool (e.g. ["discard"])
        hard_neg_weight : kept for API compatibility (no longer used for sampling weights)
        All hard-neg items are always included; remaining slots filled randomly from others.
        """
        self.feats  = np.stack([it["feat"] for it in items]).astype(np.float32)
        self.labels = np.array([it["label_id"] for it in items])
        self.focus_id = LABEL2ID[focus_label]
        self.neg_oversample = neg_oversample

        self.pos_idx = np.where(self.labels == self.focus_id)[0]
        self.neg_idx = np.where(self.labels != self.focus_id)[0]
        assert len(self.pos_idx) >= 2, "Need ≥2 positive samples for triplets"

        # Separate hard negatives (always included) from soft negatives (sampled)
        if hard_neg_labels:
            hard_ids = {LABEL2ID[l] for l in hard_neg_labels if l in LABEL2ID}
            self.hard_neg_idx = np.array(
                [idx for idx in self.neg_idx if self.labels[idx] in hard_ids])
            self.soft_neg_idx = np.array(
                [idx for idx in self.neg_idx if self.labels[idx] not in hard_ids])
        else:
            self.hard_neg_idx = np.array([], dtype=int)
            self.soft_neg_idx = self.neg_idx

        n_hard = len(self.hard_neg_idx)
        n_soft_slots = max(0, neg_oversample - n_hard)
        print(f"  TripletSampler: {len(self.pos_idx)} pos | "
              f"{n_hard} guaranteed hard negs | "
              f"{n_soft_slots} random soft negs per anchor")

    def iter_batches(self, batch_size: int):
        """Yield (anchors, positives, neg_pool) as numpy arrays.
        neg_pool = [all hard negs] + [random soft negs] for every anchor.
        """
        order = np.random.permutation(len(self.pos_idx))
        n_hard      = len(self.hard_neg_idx)
        n_soft_slots = max(1, self.neg_oversample - n_hard)

        for start in range(0, len(order), batch_size):
            a_idx = self.pos_idx[order[start:start + batch_size]]
            p_idx = np.array([
                random.choice(self.pos_idx[self.pos_idx != ai].tolist())
                for ai in a_idx
            ])

            # Soft negatives: sample randomly per anchor
            k_soft = min(n_soft_slots, len(self.soft_neg_idx))
            soft = np.stack([
                np.random.choice(self.soft_neg_idx, size=k_soft, replace=False)
                for _ in a_idx
            ])  # (B, k_soft)

            if n_hard > 0:
                # Hard negatives: same set for all anchors in batch (broadcast)
                hard = np.tile(self.hard_neg_idx, (len(a_idx), 1))  # (B, n_hard)
                n_idx = np.concatenate([hard, soft], axis=1)         # (B, K)
            else:
                n_idx = soft

            yield (self.feats[a_idx],    # (B, D)
                   self.feats[p_idx],    # (B, D)
                   self.feats[n_idx])    # (B, K, D)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_projection(train_items, in_dim: int, out_dim: int = 256,
                     epochs: int = 150, lr: float = 3e-3, margin: float = 0.3,
                     neg_oversample: int = 16, batch_size: int = 16,
                     focus_label: str = "split",
                     hard_neg_labels: list = None, hard_neg_weight: float = 5.0):
    """
    Train a linear projection with batch-hard triplet loss.
    Uses manual numpy batching (torch.tensor copies) to avoid system-torch
    NumPy ABI incompatibility with DataLoader/from_numpy.
    Returns the trained ProjectionHead (on CPU).
    """
    sampler = TripletSampler(train_items, focus_label=focus_label,
                              neg_oversample=neg_oversample,
                              hard_neg_labels=hard_neg_labels,
                              hard_neg_weight=hard_neg_weight)

    model = ProjectionHead(in_dim, out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for a_np, p_np, n_np in sampler.iter_batches(batch_size):
            # torch.tensor() copies data — no numpy ABI bridge needed
            anchors  = torch.tensor(a_np, dtype=torch.float32).to(DEVICE)
            positives= torch.tensor(p_np, dtype=torch.float32).to(DEVICE)
            neg_pool = torch.tensor(n_np, dtype=torch.float32).to(DEVICE)

            z_a = model(anchors)
            z_p = model(positives)
            B, K, D = neg_pool.shape
            z_n_all = model(neg_pool.view(B * K, D)).view(B, K, -1)

            # batch-hard negative
            sim_an  = (z_a.unsqueeze(1) * z_n_all).sum(-1)   # (B, K)
            z_n     = z_n_all[torch.arange(B), sim_an.argmax(dim=1)]

            loss = F.relu(margin - (z_a * z_p).sum(-1) + (z_a * z_n).sum(-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={total_loss/n_batches:.4f}"
                  f"  lr={scheduler.get_last_lr()[0]:.5f}")

    return model.cpu()


# --------------------------------------------------------------------------- #
# V2 — joint CE + triplet training for SplitDetector
# --------------------------------------------------------------------------- #

def split_triplet_loss(embeddings, labels, focus_id, margin=0.3):
    """Hard-negative triplet loss for the focus class."""
    focus_mask = (labels == focus_id)
    n_focus = focus_mask.sum().item()
    if n_focus < 2:
        return torch.tensor(0.0, device=embeddings.device)

    focus_emb = embeddings[focus_mask]
    other_emb = embeddings[~focus_mask]
    if other_emb.shape[0] == 0:
        return torch.tensor(0.0, device=embeddings.device)

    P = focus_emb.shape[0]
    # Random positive (offset to avoid self-pairing)
    offset = torch.randint(1, P, (P,), device=embeddings.device)
    perm = (torch.arange(P, device=embeddings.device) + offset) % P
    pos_emb = focus_emb[perm]

    # Hard negative: most similar non-focus
    neg_sims = focus_emb @ other_emb.T
    hard_neg_idx = neg_sims.argmax(dim=1)
    neg_emb = other_emb[hard_neg_idx]

    pos_sim = (focus_emb * pos_emb).sum(dim=1)
    neg_sim = (focus_emb * neg_emb).sum(dim=1)
    return F.relu(margin - pos_sim + neg_sim).mean()


def train_detector(train_items, in_dim, embed_dim=256,
                   n_classes=len(ALL_LABELS),
                   epochs=300, lr=3e-3, margin=0.3, batch_size=64,
                   triplet_weight=0.5, feat_dropout=0.15,
                   focus_label="split"):
    """
    Train SplitDetector with joint CE + triplet loss.

    - CE loss (class-weighted) teaches multi-class boundaries
    - Triplet loss (split-focused, hard negative) refines the split cluster
    - Balanced sampling: 25% split per batch
    - Feature dropout: online augmentation on input features
    """
    feats = np.stack([it["feat"] for it in train_items]).astype(np.float32)
    labels = np.array([it["label_id"] for it in train_items])
    focus_id = LABEL2ID[focus_label]

    # No global class weights — balanced sampling already ensures 25% split
    # per batch. Global weights hurt because ultra-rare classes (wave, clean
    # hand) dominate the loss.  Only boost split in the CE if needed.
    class_weights = None

    model = SplitDetector(in_dim, embed_dim, n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    focus_idx = np.where(labels == focus_id)[0]
    other_idx = np.where(labels != focus_id)[0]
    n_focus_per_batch = max(4, batch_size // 4)
    n_other_per_batch = batch_size - n_focus_per_batch
    n_batches = max(4, len(focus_idx) * 2 // n_focus_per_batch)

    print(f"  SplitDetector: {in_dim}→{embed_dim}  {n_classes} classes")
    print(f"  Balanced batches: {n_focus_per_batch} focus + "
          f"{n_other_per_batch} other × {n_batches}/epoch")
    print(f"  Class weights: None (balanced sampling handles imbalance)")

    model.train()
    for epoch in range(epochs):
        epoch_ce = 0.0
        epoch_tri = 0.0
        n = 0

        for _ in range(n_batches):
            f_idx = np.random.choice(
                focus_idx, size=n_focus_per_batch,
                replace=len(focus_idx) < n_focus_per_batch)
            o_idx = np.random.choice(
                other_idx, size=n_other_per_batch,
                replace=len(other_idx) < n_other_per_batch)
            b_idx = np.concatenate([f_idx, o_idx])
            np.random.shuffle(b_idx)

            b_feats = torch.tensor(feats[b_idx],
                                   dtype=torch.float32).to(DEVICE)
            b_labels = torch.tensor(labels[b_idx],
                                    dtype=torch.long).to(DEVICE)

            # Feature dropout augmentation
            if feat_dropout > 0:
                mask = (torch.rand_like(b_feats) > feat_dropout).float()
                b_feats = b_feats * mask / (1 - feat_dropout)

            embeddings, logits = model(b_feats)

            ce_loss = F.cross_entropy(logits, b_labels, weight=class_weights)
            tri_loss = split_triplet_loss(
                embeddings, b_labels, focus_id, margin)

            loss = ce_loss + triplet_weight * tri_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_tri += tri_loss.item()
            n += 1

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"CE={epoch_ce/n:.4f}  tri={epoch_tri/n:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    return model.cpu()


@torch.no_grad()
def project_items(model, items):
    """Project items through the model's encoder, return (N, embed_dim) numpy."""
    model.eval()
    feats_np = np.stack([it["feat"] for it in items]).astype(np.float32)
    out = []
    bs = 512
    for i in range(0, len(feats_np), bs):
        chunk = torch.tensor(feats_np[i:i+bs], dtype=torch.float32)
        # .embed() works for both ProjectionHead and SplitDetector
        # .tolist() avoids torch→numpy C ABI bridge (system-torch NumPy 1.x issue)
        out.append(np.array(model.embed(chunk).tolist(), dtype=np.float32))
    return np.concatenate(out, axis=0)

