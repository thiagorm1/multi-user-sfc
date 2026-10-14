"""Prepare new chains while retaining the old ones; commit on an unchanged network."""

import copy
import hashlib
import math
import pickle
from dataclasses import dataclass


def fingerprint(net):
    return hashlib.sha256(pickle.dumps(net, protocol=5)).hexdigest()


@dataclass
class PreparedHandover:
    baseline_hash: str
    candidate: object
    old_ids: tuple
    new_ids: tuple
    latencies_ms: tuple
    committed: bool = False

    def commit(self, current):
        from muar_sfc.core.scenarios.muvr_reference import audit

        if self.committed:
            raise ValueError("Handover was already committed")
        if fingerprint(current) != self.baseline_hash:
            raise ValueError("Network changed; prepare a fresh handover")
        replacement = copy.deepcopy(self.candidate)
        # New chains are already reserved before old chains are released.
        for old_id in self.old_ids:
            replacement.undeploy_sfc(old_id)
        audit(replacement)
        self.committed = True
        return replacement


def prepare_handover(net, old_ids, new_batch, admitter, strategy="mappo"):
    """Return a prepared snapshot, or None; the active network is never mutated.

    New SFC IDs must differ from old IDs. Both versions must fit simultaneously.
    The caller commits the returned network on its orchestration thread. This
    models resource-safe cutover, not packet switching or renderer continuity.
    """
    from muar_sfc.core.scenarios.muvr_reference import audit

    old_ids = tuple(old_ids)
    new_ids = tuple(s.id for s in new_batch)
    if not old_ids or any(i not in net.sfc_dict for i in old_ids):
        raise ValueError("Old chains must be active")
    if not new_ids or len(set(new_ids)) != len(new_ids) or any(i in net.sfc_dict for i in new_ids):
        raise ValueError("Replacement chains need new unique IDs")
    baseline = fingerprint(net)
    snapshot = copy.deepcopy(net)
    candidate, latencies, _ = admitter(snapshot, copy.deepcopy(new_batch), strategy)
    if candidate is None:
        return None
    if any(i not in candidate.sfc_dict for i in (*old_ids, *new_ids)):
        raise ValueError("Prepared network must retain old and new chains")
    if len(latencies) != len(new_batch) or any(
        not math.isfinite(latency) or latency < 0 or latency > sfc.get_latency_request()
        for latency, sfc in zip(latencies, new_batch, strict=True)
    ):
        raise ValueError("Replacement violates SLA")
    audit(candidate)
    return PreparedHandover(baseline, candidate, old_ids, new_ids, tuple(latencies))
