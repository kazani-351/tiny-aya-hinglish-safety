#!/bin/sh
# Fail if a ref about to be pushed does not contain every OpenTimestamps-stamped commit.
# Stamped commits are listed by the proofs/commit-<sha>.sha files. A rebase, amend, squash or
# filter-repo that rewrites any of them changes its SHA, so the stamp would no longer match
# anything public. Usage: proofs/check_stamped_shas.sh <ref-or-sha>   (exit 0 = all present)
# Called by .git/hooks/pre-push. Log: ~/.claude/logs/lexsi-prepush-<date>.log
ref="${1:-HEAD}"
root="$(git rev-parse --show-toplevel)" || exit 2
log="$HOME/.claude/logs/lexsi-prepush-$(date +%F).log"
mkdir -p "$(dirname "$log")"
tip="$(git rev-parse --verify "$ref^{commit}" 2>/dev/null)" || { echo "FAIL: cannot resolve $ref" | tee -a "$log" >&2; exit 2; }
n=0; bad=0
for f in "$root"/proofs/commit-*.sha; do
  [ -e "$f" ] || continue
  sha="$(tr -d ' \n' < "$f")"; n=$((n + 1))
  if git merge-base --is-ancestor "$sha" "$tip" 2>/dev/null; then
    echo "ok   $sha is in $ref" >> "$log"
  else
    echo "FAIL stamped commit $sha is NOT in $ref ($tip): history was rewritten or the ref is wrong" | tee -a "$log" >&2
    bad=$((bad + 1))
  fi
done
[ "$n" -gt 0 ] || { echo "FAIL: no proofs/commit-*.sha files found" | tee -a "$log" >&2; exit 2; }
echo "$(date '+%F %T') checked $n stamped commits against $ref: $bad missing" >> "$log"
[ "$bad" -eq 0 ]
