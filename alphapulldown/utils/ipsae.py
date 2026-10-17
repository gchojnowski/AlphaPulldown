"""
ipSAE (Dunbrack 2025, https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2)
from a model's PAE matrix and chain lengths: the "ipSAE" column of the
reference script (https://github.com/DunbrackLab/IPSAE, ipsae.py v4, MIT), i.e.
the d0res variant, with the same PAE cutoff and d0 formula.

For chains a -> b: each residue i of a is scored against the residues j of b
with PAE(i, j) < cutoff, using a pTM-like term with d0 from their number;
ipSAE(a -> b) is the best residue's score, and ipSAE(a, b) the larger of
a -> b and b -> a. Only the PAE is needed (no coordinates); the reference
script's other outputs are not reproduced, except two more per-pair scores
computed the same way as there:

- LIS (Kim et al. 2024): mean of (12 - PAE) / 12 over the pair's inter-chain
  PAE values below 12 A, averaged over a -> b and b -> a. PAE only.
- pDockQ2 (Zhu et al. 2023): from residue pairs within 8 A (C-beta, C-alpha for
  glycine), their mean pTM-like PAE term (d0 = 10) and the mean pLDDT of the
  residues involved; the larger of a -> b and b -> a. Needs coordinates.
"""
import math
from typing import Dict, Sequence

import numpy as np

PAE_CUTOFF = 10.0      # Angstrom; the authors' recommendation for AlphaFold2


def _d0(n: np.ndarray) -> np.ndarray:
    """Yang & Skolnick d0 with the reference script's floor (n >= 26, d0 >= 1)."""
    n = np.maximum(26.0, np.asarray(n, dtype=float))
    return np.maximum(1.0, 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae_asym(pae: np.ndarray, rows: np.ndarray, cols: np.ndarray,
               cutoff: float = PAE_CUTOFF) -> float:
    """ipSAE for aligned residues `rows` (chain a) scored on `cols` (chain b)."""
    block = pae[np.ix_(rows, cols)]
    valid = block < cutoff
    n = valid.sum(axis=1)
    if not n.any():
        return 0.0
    d0 = _d0(n)[:, None]
    ptm = 1.0 / (1.0 + (block / d0) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_residue = np.where(n > 0, (ptm * valid).sum(axis=1) / np.maximum(n, 1), 0.0)
    return float(per_residue.max())


LIS_CUTOFF = 12.0      # Angstrom (Kim et al.)
CONTACT = 8.0          # Angstrom, C-beta distance for pDockQ2


def lis_asym(pae: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    block = pae[np.ix_(rows, cols)]
    valid = block[block < LIS_CUTOFF]
    return float(np.mean((LIS_CUTOFF - valid) / LIS_CUTOFF)) if valid.size else 0.0


def pdockq2_asym(pae: np.ndarray, dist: np.ndarray, plddt: np.ndarray,
                 rows: np.ndarray, cols: np.ndarray) -> float:
    close = dist[np.ix_(rows, cols)] <= CONTACT
    npairs = int(close.sum())
    if not npairs:
        return 0.0
    mean_ptm = float((1.0 / (1.0 + (pae[np.ix_(rows, cols)][close] / 10.0) ** 2)).sum()) / npairs
    involved = np.concatenate([rows[close.any(axis=1)], cols[close.any(axis=0)]])
    x = float(plddt[involved].mean()) * mean_ptm
    return 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005


def _chains(chain_lengths: Sequence[int], n: int):
    ends = np.cumsum(chain_lengths)
    if not len(ends) or ends[-1] != n:
        raise ValueError(f"chain lengths {list(chain_lengths)} do not match {n} residues")
    starts = ends - np.asarray(chain_lengths)
    idx = [np.arange(s, e) for s, e in zip(starts, ends)]
    labels = [chr(ord("A") + k) if k < 26 else f"C{k}" for k in range(len(idx))]
    return idx, labels


def chain_pair_ipsae(pae, chain_lengths: Sequence[int],
                     cutoff: float = PAE_CUTOFF) -> Dict[str, float]:
    """{"A-B": ipSAE, ...} for every chain pair, chains A, B, ... in input order."""
    return {k: v["ipsae"] for k, v in chain_pair_scores(pae, chain_lengths, cutoff=cutoff).items()}


def chain_pair_scores(pae, chain_lengths: Sequence[int], cb_coords=None, plddt=None,
                      cutoff: float = PAE_CUTOFF) -> Dict[str, Dict[str, float]]:
    """
    {"A-B": {"ipsae": ..., "lis": ..., "pdockq2": ...}, ...} for every chain
    pair. pdockq2 only with cb_coords (N x 3: C-beta, C-alpha for glycine) and
    per-residue plddt.
    """
    pae = np.asarray(pae, dtype=float)
    idx, labels = _chains(chain_lengths, pae.shape[0])
    dist = None
    n = pae.shape[0]
    if cb_coords is not None and plddt is not None:
        # Prediction outputs may carry padding after the last residue.
        xyz = np.asarray(cb_coords, dtype=float)[:n]
        plddt = np.asarray(plddt, dtype=float)[:n]
        if len(xyz) == n and len(plddt) == n:      # otherwise no pDockQ2, the rest still
            dist = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2))
    out = {}
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            ra, rb = idx[a], idx[b]
            scores = {
                "ipsae": round(max(ipsae_asym(pae, ra, rb, cutoff), ipsae_asym(pae, rb, ra, cutoff)), 4),
                "lis": round((lis_asym(pae, ra, rb) + lis_asym(pae, rb, ra)) / 2, 4),
            }
            if dist is not None:
                scores["pdockq2"] = round(max(pdockq2_asym(pae, dist, plddt, ra, rb),
                                              pdockq2_asym(pae, dist, plddt, rb, ra)), 4)
            out[f"{labels[a]}-{labels[b]}"] = scores
    return out


def cb_coordinates(atom_positions, aatype) -> np.ndarray:
    """C-beta per residue (C-alpha for glycine) from AlphaFold atom37 positions."""
    from alphafold.common import residue_constants as rc
    pos = np.asarray(atom_positions)
    cb = pos[:, rc.atom_order["CB"], :].copy()
    gly = np.asarray(aatype) == rc.restype_order["G"]
    cb[gly] = pos[gly, rc.atom_order["CA"], :]
    return cb
