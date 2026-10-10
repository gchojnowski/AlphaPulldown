"""
Template structures fetched from RCSB on demand, instead of a local PDB mmCIF
mirror (~240 GB). Only the hits' entries are downloaded — typically a few
dozen per protein — into a cache directory that later runs reuse.

Used by create_individual_features.py with --fetch_templates (and always for
--use_afdb, whose template search already downloaded its mmCIFs).
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests
from absl import logging

from alphafold.data import parsers, templates

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def fetch_mmcifs(pdb_ids: Iterable[str], mmcif_dir: str, timeout: int = 120,
                 retries: int = 2, workers: int = 8) -> set:
    """
    Downloads <pdb_id>.cif for each id not already in mmcif_dir. Returns the
    ids now present. A failed download (network, obsolete entry, timeout on a
    very large structure) is logged and skipped rather than raised: losing one
    template is better than losing the protein.

    Files are written to a temporary name and renamed, so jobs sharing a cache
    directory never see a partial file.
    """
    Path(mmcif_dir).mkdir(parents=True, exist_ok=True)
    ids = list(dict.fromkeys(p.lower() for p in pdb_ids))

    def one(pdb_id: str) -> Optional[str]:
        path = Path(mmcif_dir) / f"{pdb_id}.cif"
        if path.is_file() and path.stat().st_size > 0:
            return pdb_id
        url = RCSB_URL.format(pdb_id=pdb_id)
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 404:
                    logging.warning("Template %s: not found at RCSB (obsolete?), skipped", pdb_id)
                    return None
                r.raise_for_status()
                fd, tmp = tempfile.mkstemp(dir=mmcif_dir, prefix=f".{pdb_id}.", suffix=".part")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(r.content)
                os.replace(tmp, path)
                logging.info("Downloaded mmCIF %s", url)
                return pdb_id
            except (requests.RequestException, OSError) as exc:
                logging.warning("Template %s: download attempt %d failed: %s",
                                pdb_id, attempt + 1, exc)
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {p for p in pool.map(one, ids) if p}


class OnDemandHmmsearchHitFeaturizer(templates.HmmsearchHitFeaturizer):
    """
    HmmsearchHitFeaturizer whose mmcif_dir is a download cache: before
    featurizing, the hits' entries are fetched from RCSB. Hits are taken in
    order until max_fetch distinct entries — the featurizer keeps at most
    max_hits (20) templates anyway — and hits whose entry could not be fetched
    are dropped, since AlphaFold would otherwise fail on the missing file.
    """

    def __init__(self, mmcif_dir: str, max_template_date: str, max_hits: int,
                 kalign_binary_path: str, release_dates_path: Optional[str],
                 obsolete_pdbs_path: Optional[str], strict_error_check: bool = False,
                 max_fetch: int = 100):
        # The base __init__ refuses a directory without any .cif files, which is
        # exactly what an empty cache is. Everything else it sets up is repeated
        # here unchanged.
        Path(mmcif_dir).mkdir(parents=True, exist_ok=True)
        self._mmcif_dir = mmcif_dir
        try:
            self._max_template_date = templates.datetime.datetime.strptime(
                max_template_date, "%Y-%m-%d")
        except (TypeError, ValueError) as e:
            raise ValueError("max_template_date must be set and have format YYYY-MM-DD.") from e
        self._max_hits = max_hits
        self._kalign_binary_path = kalign_binary_path
        self._strict_error_check = strict_error_check
        self._release_dates = (templates._parse_release_dates(release_dates_path)
                               if release_dates_path else {})
        self._obsolete_pdbs = (templates._parse_obsolete(obsolete_pdbs_path)
                               if obsolete_pdbs_path and os.path.isfile(obsolete_pdbs_path) else {})
        self._max_fetch = max_fetch
        logging.info("Template mmCIFs are fetched on demand into %s", mmcif_dir)

    def _entry(self, hit: parsers.TemplateHit) -> Optional[str]:
        """The PDB entry AlphaFold will read for this hit (after obsolete mapping)."""
        try:
            pdb_id, _ = templates._get_pdb_id_and_chain(hit)
        except ValueError:
            return None  # the base class reports malformed hit names itself
        if pdb_id in self._obsolete_pdbs and pdb_id not in self._release_dates:
            return self._obsolete_pdbs[pdb_id]  # None = removed from the PDB
        return pdb_id

    def get_templates(self, query_sequence: str,
                      hits: Sequence[parsers.TemplateHit]) -> templates.TemplateSearchResult:
        hits = list(hits)
        if hits and hits[0].sum_probs is not None:      # same order the base class uses
            hits.sort(key=lambda h: h.sum_probs, reverse=True)

        wanted: list = []
        selected = []
        for hit in hits:
            entry = self._entry(hit)
            if entry and entry not in wanted:
                if len(wanted) >= self._max_fetch:
                    continue
                wanted.append(entry)
            selected.append(hit)

        available = fetch_mmcifs(wanted, self._mmcif_dir)
        logging.info("Templates: %d hits in %d entries, %d fetched or cached",
                     len(selected), len(wanted), len(available))
        # Keep hits whose file is there; keep unresolvable/obsolete ones too —
        # the base class skips those itself with a proper warning.
        kept = [h for h in selected if self._entry(h) is None or self._entry(h) in available]
        return super().get_templates(query_sequence=query_sequence, hits=kept)
