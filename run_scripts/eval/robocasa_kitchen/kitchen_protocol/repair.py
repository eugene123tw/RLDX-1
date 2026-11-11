#!/usr/bin/env python
"""Scan a finished RoboCasa Kitchen run, quarantine contaminated episodes, and
re-run exactly those slots with one environment per process.

Three subcommands, meant to be run in order:

  scan    judge every mp4, write ``swap_manifest.json`` and append to
          ``swap_cycles.json``. Read-only with respect to the run tree.
  apply   move the flagged mp4s into ``_quarantine_swap/`` and run ``SUBMIT_CMD``
          to re-evaluate the shards holding them. Dry-run unless ``--submit``.
  report  print the before/after success rate and every outcome that moved.

The run directory is the one ``eval_robocasa.sh`` writes::

    <run dir>/<Task>/robocasa_..._env<NN>-episode_<K>-{success,failure}.mp4

Configure with environment variables or flags:

  GR00T_ROOT  this repo, for the task ordering                  (--gr00t-root)
  SUBMIT_CMD  shell template used by ``apply --submit``         (--submit-cmd)

``SUBMIT_CMD`` is formatted with ``{array}``, the shard indices to re-run.
``swap_chain.sh`` passes one that re-runs those shards in place, and is the
usual way to run all three steps.

Run it with the robocasa venv, which has numpy and cv2::

    cd $RLDX/run_scripts/eval/robocasa_kitchen
    PYTHONPATH=. robocasa_uv/.venv/bin/python \
        -m kitchen_protocol.repair scan <run dir>
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess

from .detect import episode_signature, classify

MP4 = re.compile(r"env(\d+)-episode_(\d+)-(success|failure)\.mp4$")
#: Directories inside a run that are not tasks.
NON_TASK = {"_launcher_logs", "_quarantine_swap"}


def run_dir(args):
    d = args.run.rstrip("/")
    if not os.path.isdir(d):
        raise SystemExit(f"no such run directory: {d}")
    return d


def task_paths(root):
    """Task directories. Ours are bare names, so non-task entries are excluded."""
    return sorted(d for d in glob.glob(f"{root}/*")
                  if os.path.isdir(d) and os.path.basename(d) not in NON_TASK)


def task_order(args):
    """Task name -> array index, read from the eval script's own task list.

    The order the eval script iterates is the only definition of the array
    index, so it is parsed rather than duplicated here. This launcher gives one
    array task several tasks, so the index is the shard: TASK_NAMES fixes the
    order and N_GPUS the shares.
    """
    root = args.gr00t_root or os.environ.get("GR00T_ROOT")
    if not root:
        raise SystemExit("set GR00T_ROOT or pass --gr00t-root")
    txt = open(f"{root}/run_scripts/eval/robocasa_kitchen/eval_robocasa.sh").read()
    block = re.search(r"^TASK_NAMES=\((.*?)^\)", txt, re.S | re.M)
    names = re.findall(r'"([A-Za-z0-9_]+)"', block.group(1)) if block else []
    if len(names) != 24:
        raise SystemExit(f"parsed {len(names)} kitchen tasks, expected 24")
    base, extra = divmod(len(names), args.n_gpus)
    order = {}
    for shard in range(args.n_gpus):
        lo = shard * (base + 1) if shard < extra else extra * (base + 1) + (shard - extra) * base
        hi = lo + base + (1 if shard < extra else 0)
        for n in names[lo:hi]:
            order[n] = shard
    return order


# --------------------------------------------------------------------------- scan

def scan(args):
    root = run_dir(args)
    rows, undecodable = [], []
    for td in task_paths(root):
        task = os.path.basename(td)
        for p in sorted(glob.glob(f"{td}/*.mp4")):
            m = MP4.search(p)
            if not m:
                continue
            sig = episode_signature(p)
            reason = classify(sig)
            if reason is None:
                continue
            stats = sig if isinstance(sig, dict) else {"med": -1, "gap": 0.0, "lag1m": 0.0}
            rec = {
                "task": task, "env": int(m[1]), "ep": int(m[2]),
                "out_before": m[3], "mp4": os.path.basename(p),
                "reason": reason, **stats,
            }
            # `undecodable` is recorded but never repaired. The success label is written
            # by the environment into the filename, not read back from the video, so a
            # damaged recording is not evidence of a damaged observation -- re-running
            # discards a valid label and redraws an unseeded noise sample. Measured:
            # swap-alt/sustained-nonphys flipped success->failure 0 times in 46 re-runs,
            # undecodable flipped it 13 times in 44.
            (undecodable if reason == "undecodable" else rows).append(rec)

    json.dump({"run": args.run, "contaminated": rows, "undecodable": undecodable},
              open(f"{root}/swap_manifest.json", "w"), indent=1)
    cycles = _append_cycle(root, rows)

    print(f"[scan] {len(rows)} contaminated (cycle {cycles[-1]['n']})")
    if undecodable:
        print(f"[scan] {len(undecodable)} undecodable -- recorded only, original label kept")
    for r in rows:
        print(f"   {r['task'].replace('_PandaOmron_Env',''):<24} "
              f"env{r['env']}ep{r['ep']:<3} {r['out_before']:<8} "
              f"med={r['med']:>6} gap={r['gap']:>6} lag1m={r['lag1m']:>6} [{r['reason']}]")
    return rows


def _append_cycle(root, rows):
    """Maintain the cycle history that before/after reporting reads.

    Two rules keep it honest. If a previous tree was repaired before this file
    existed, cycle 1 is rebuilt from the quarantined originals -- their
    filenames carry (task, env, episode, original outcome), which is everything
    the 'before' column needs. Without that rebuild the first scan of such a
    tree writes itself as cycle 1 and the original outcomes are lost. And if the
    last recorded cycle has not been re-run yet, this scan replaces it instead
    of appending, so repeated scans without an intervening repair do not inflate
    the cycle count.
    """
    path = f"{root}/swap_cycles.json"
    cycles = json.load(open(path))["cycles"] if os.path.exists(path) else []

    if not cycles and os.path.isdir(f"{root}/_quarantine_swap"):
        prior = []
        for qp in glob.glob(f"{root}/_quarantine_swap/*/*.mp4"):
            m = MP4.search(qp)
            if m:
                prior.append({"task": os.path.basename(os.path.dirname(qp)),
                              "env": int(m[1]), "ep": int(m[2]), "out_before": m[3],
                              "mp4": os.path.basename(qp),
                              "reason": "reconstructed-from-quarantine"})
        if prior:
            cycles = [{"n": 1, "contaminated": prior}]
            print(f"[scan] rebuilt cycle 1 ({len(prior)} entries) from quarantine")

    def refilled(recs):
        return all(glob.glob(f"{root}/{r['task']}/*env{r['env']:02d}-episode_{r['ep']}-*.mp4")
                   for r in recs) if recs else True

    if cycles and not refilled(cycles[-1]["contaminated"]):
        cycles[-1] = {"n": cycles[-1]["n"], "contaminated": rows}
    else:
        cycles.append({"n": len(cycles) + 1, "contaminated": rows})
    json.dump({"cycles": cycles}, open(path, "w"), indent=1)
    return cycles


# -------------------------------------------------------------------------- apply

def apply(args):
    root = run_dir(args)
    man = json.load(open(f"{root}/swap_manifest.json"))
    rows = man["contaminated"]
    if not rows:
        print("[apply] nothing flagged; tree has converged")
        return

    # Checked before anything moves: quarantined episodes with nothing to
    # re-evaluate them leave the tree short.
    template = args.submit_cmd or os.environ.get("SUBMIT_CMD")
    if args.submit and not template:
        raise SystemExit("--submit needs SUBMIT_CMD or --submit-cmd; swap_chain.sh sets one")

    quarantine = f"{root}/_quarantine_swap"
    for r in rows:
        src = f"{root}/{r['task']}/{r['mp4']}"
        if not os.path.exists(src):
            continue
        dst = f"{quarantine}/{r['task']}/{r['mp4']}"
        print(f"[{'move' if args.submit else 'would-move'}] {r['task']}/{r['mp4']}")
        if args.submit:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    order = task_order(args)
    idx = sorted(order[t] for t in {r["task"] for r in rows})
    print(f"[apply] {len(rows)} episodes across {len(idx)} tasks, array={idx}")
    if not template:
        print("[dry-run] no SUBMIT_CMD set; nothing would re-evaluate them")
        return
    cmd = template.format(array=",".join(map(str, idx)))
    if args.submit:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
    else:
        print("[dry-run] would run:\n  " + cmd)


# ------------------------------------------------------------------------- report

def report(args):
    root = run_dir(args)
    path = f"{root}/swap_cycles.json"
    if not os.path.exists(path):
        raise SystemExit("no swap_cycles.json -- run scan first")
    cycles = json.load(open(path))["cycles"]

    current = {}
    for td in task_paths(root):
        task = os.path.basename(td)
        for p in glob.glob(f"{td}/*.mp4"):
            m = MP4.search(p)
            if m:
                current[(task, int(m[1]), int(m[2]))] = m[3]

    succ = sum(v == "success" for v in current.values())
    total = len(current)
    print(f"[report] {root}")

    # Walk the cycles backwards, undoing each repair to recover the earlier rate.
    rates = [(succ, total)]
    s = succ
    for c in reversed(cycles):
        for r in c["contaminated"]:
            now = current.get((r["task"], r["env"], r["ep"]))
            if now is None:
                continue
            s += (r["out_before"] == "success") - (now == "success")
        rates.append((s, total))
    rates.reverse()

    labels = ["original"] + [f"after cycle {c['n']}" for c in cycles]
    for label, (a, b) in zip(labels, rates):
        print(f"  {label:<16} {a}/{b} = {100 * a / max(b, 1):.2f}%")

    pending = [r for c in cycles for r in c["contaminated"]
               if (r["task"], r["env"], r["ep"]) not in current]
    if pending:
        print(f"  !! {len(pending)} slots still empty -- repair not finished")

    for c in cycles:
        moved = [(r, current.get((r["task"], r["env"], r["ep"])))
                 for r in c["contaminated"]]
        moved = [(r, n) for r, n in moved if n is not None and n != r["out_before"]]
        if moved:
            print(f"  cycle {c['n']}: {len(moved)} outcome(s) changed")
            for r, now in moved:
                print(f"    {r['task'].replace('_PandaOmron_Env',''):<24} "
                      f"env{r['env']}ep{r['ep']:<3} {r['out_before']} -> {now}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "apply", "report"])
    ap.add_argument("run", help="run directory eval_robocasa.sh wrote")
    ap.add_argument("--n-gpus", type=int, default=8, help="shard count the run used")
    ap.add_argument("--gr00t-root",
                    default=os.path.abspath(f"{os.path.dirname(__file__)}/../../../.."))
    ap.add_argument("--submit-cmd", default="", help="submission template (apply)")
    ap.add_argument("--submit", action="store_true", help="actually move and re-evaluate")
    args = ap.parse_args()

    if args.mode == "scan":
        scan(args)
    elif args.mode == "apply":
        apply(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
