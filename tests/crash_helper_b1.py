"""Crash victim for the B1 ledger tests. NOT a test module.

Run as
`python crash_helper_b1.py <stage> <repo> <state_dir> <baseline> <tempdir>`.

`tempdir` is passed EXPLICITLY rather than inherited: the worktree root
is derived from the process's temp directory, and a child that resolved
the shared system-wide one would create holders where a real agent loop
keeps its own.
It advances the worktree lifecycle to the requested stage, prints READY
so the parent knows the stage was reached, and then blocks forever
waiting to be killed.

WHY A SEPARATE PROCESS. Raising inside `create` is not a crash: `create`
catches `BaseException` and cleans up, so an exception-based test proves
the tidy path works and says nothing about the untidy one. A real crash
is a process that stops between two statements without running anything
afterwards, and the only honest way to produce that is to kill one.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.agent_loop import worktree  # noqa: E402


def main():
    stage, repo, state_dir, baseline, private_temp = sys.argv[1:6]
    repo, state_dir = Path(repo), Path(state_dir)
    tempfile.tempdir = private_temp
    assert worktree.runner_temp_root().parent == Path(private_temp), \
        "kurban surec paylasilan gecici koke yaziyordu"

    record = worktree.register(state_dir, repo=repo, run_id="kurgu-run-1",
                               baseline_sha=baseline)
    worktree_id = record["worktree_id"]
    holder = worktree.holder_for(worktree_id)

    if stage != "record_only":
        holder.mkdir()
    if stage in ("git_added", "ready"):
        worktree._git(repo, "worktree", "add", "--detach",       # noqa: SLF001
                      str(holder / worktree.WORKTREE_DIRNAME), baseline)
    if stage == "ready":
        from tools.agent_loop import state as state_module

        state_module.write_json_atomically(
            worktree._record_path(state_dir, worktree_id),        # noqa: SLF001
            dict(record, status=worktree.STATUS_READY),
            worktree.RECORD_SCHEMA, "calisma agaci kaydi")

    print("READY " + worktree_id, flush=True)
    while True:                       # wait to be killed
        time.sleep(3600)


if __name__ == "__main__":
    main()
