"""Closed organization-topology validation independent of HTTP and SQL."""
from __future__ import annotations

from collections import deque


class TopologyRefused(ValueError):
    """A proposed hierarchy is ambiguous, disconnected, or too permissive."""


def ordered_topology(positions, members):
    """Validate a complete topology and return parent-before-child positions.

    A tree is directional authorization data, not presentation data.  The
    validator therefore requires exactly one root, every non-root parent to
    exist, no cycles or disconnected components, unique occupied positions,
    and leaf members without monitoring capability.
    """
    position_rows = [dict(item) for item in positions]
    member_rows = [dict(item) for item in members]
    by_id = {row["id"]: row for row in position_rows}
    if len(by_id) != len(position_rows):
        raise TopologyRefused("pozisyon kimlikleri benzersiz olmali")
    roots = [row for row in position_rows if row["parent_id"] is None]
    if len(roots) != 1 or roots[0]["kind"] != "root":
        raise TopologyRefused("organizasyonun tam bir root'u olmali")
    if (not roots[0]["can_monitor_descendants"]
            or not roots[0]["protected_from_monitoring"]):
        raise TopologyRefused(
            "root tum alt agaci gorebilmeli ve goruntulenmemeli")

    children = {position_id: [] for position_id in by_id}
    for row in position_rows:
        parent_id = row["parent_id"]
        if parent_id is None:
            continue
        if row["kind"] == "root" or parent_id not in by_id:
            raise TopologyRefused("pozisyon parent baglantisi gecersiz")
        children[parent_id].append(row["id"])
        if row["kind"] == "member" and row["can_monitor_descendants"]:
            raise TopologyRefused("alt seviye uye monitor yetkisi alamaz")

    ordered = []
    queue = deque([roots[0]["id"]])
    seen = set()
    while queue:
        position_id = queue.popleft()
        if position_id in seen:
            raise TopologyRefused("organizasyon dongu iceremez")
        seen.add(position_id)
        ordered.append(by_id[position_id])
        queue.extend(children[position_id])
    if seen != set(by_id):
        raise TopologyRefused("organizasyon kopuk veya dongulu")

    occupied = set()
    subjects = set()
    for row in member_rows:
        position_id = row["position_id"]
        subject_key = (row["issuer"], row["subject"])
        if position_id not in by_id:
            raise TopologyRefused("uye pozisyonu organizasyonda yok")
        if position_id in occupied:
            raise TopologyRefused("aktif pozisyonda yalniz bir uye olabilir")
        if subject_key in subjects:
            raise TopologyRefused("kimlik organizasyonda bir kez yer alabilir")
        occupied.add(position_id)
        subjects.add(subject_key)
    return ordered, member_rows
