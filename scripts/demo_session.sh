#!/usr/bin/env bash
# Scripted terminal demo (recorded with asciinema): tests, cost tables, decision map, results tables. ~2 minutes.
cd "$(dirname "$0")/.."
say_run() { printf '\n\033[1;34m$ %s\033[0m\n' "$1"; sleep 1.2; eval "$1"; sleep 2; }
printf '\033[1mKVPick: KV-cache compression kernel + send-or-recompute study (demo)\033[0m\n'; sleep 2
say_run "git log --oneline -1 2>/dev/null || echo '(repo root: '\$PWD')'"
say_run "tree -L 1 -d . 2>/dev/null || ls -d */"
say_run "make test"
say_run "python scripts/build_cost_tables.py --kernels results/kernels_a40.json | head -4 | cut -c1-160"
say_run "python scripts/decision_map.py | tail -9"
say_run "python scripts/make_tables.py > /dev/null && sed -n '1,12p' docs/tables.md | cut -c1-150"
say_run "grep -A5 '### Recompute vs KV4' docs/tables.md"
say_run "grep -A9 '### Quality' docs/tables.md | cut -c1-150"
say_run "ls figures/"
printf '\n\033[1mDone. Every number above is read from results/*.json; see README.md and DECISIONS.md.\033[0m\n'; sleep 3
