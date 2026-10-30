#!/bin/sh
# git フックを有効化する(clone 後に 1 回実行)
set -e
root="$(git rev-parse --show-toplevel)"
chmod +x "$root"/scripts/hooks/* "$root"/scripts/check_no_private_data.py
git -C "$root" config core.hooksPath scripts/hooks
echo "hooks enabled: core.hooksPath=scripts/hooks"
