#!/bin/sh
# Publish gate for this repo: run before pushing main anywhere public. Exit 0 only if every check passes.
# Automatic checks run against HEAD; MANUAL items must be confirmed by kazani and recorded in HANDOFF.md.
# Usage: proofs/publish_gate.sh      Log: ~/.claude/logs/lexsi-publish-gate-<date>.log
cd "$(git rev-parse --show-toplevel)" || exit 2
log="$HOME/.claude/logs/lexsi-publish-gate-$(date +%F).log"; mkdir -p "$(dirname "$log")"
fail=0
ok()   { echo "PASS  $1" | tee -a "$log"; }
bad()  { echo "FAIL  $1" | tee -a "$log"; fail=1; }
echo "== $(date '+%F %T') gate on $(git rev-parse --short HEAD)" >> "$log"

# 1. Stamped commits still in history (OpenTimestamps proofs stay valid)
proofs/check_stamped_shas.sh HEAD >/dev/null 2>&1 && ok "stamped commits are ancestors of HEAD" || bad "a stamped commit is missing from HEAD"

# 2. Each commit proof is confirmed in a Bitcoin block (needs the ots client on PATH)
if command -v ots >/dev/null 2>&1; then
  for f in proofs/commit-*.sha.ots; do
    ots info "$f" 2>/dev/null | grep -q "BitcoinBlockHeaderAttestation" && ok "OTS confirmed: $f" || bad "OTS still pending: $f (run: ots upgrade $f)"
  done
else
  bad "ots client not on PATH: cannot confirm OTS proofs"
fi

# 3. No placeholder left in the files that ship
n=$(git grep -c "\[K:" HEAD -- README.md drafts/MODEL_CARD.md 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
[ "$n" -eq 0 ] && ok "no [K: ...] placeholders in README.md / MODEL_CARD.md" || bad "$n [K: ...] placeholders left in README.md / MODEL_CARD.md"

# 4. No absolute local paths in tracked files at HEAD
p=$(git grep -l -E "/Use[r]s/|/priv[a]te/tmp|/var/f[o]lders|/h[o]me/[a-z]" HEAD 2>/dev/null | wc -l | tr -d ' ')
[ "$p" -eq 0 ] && ok "no absolute local paths at HEAD" || bad "$p files at HEAD contain absolute local paths"

# 5. Referral link and identity strings not at HEAD. The patterns live OUTSIDE the repo, one regex per line,
#    in ~/.config/lexsi/gate-patterns (chmod 600), so this script never contains them.
#    An empty pattern file would make `git grep -f` match nothing and PASS silently, so require >= 1 real line.
#    Override the location with GATE_PATTERNS=/path for another machine or CI.
pat="${GATE_PATTERNS:-$HOME/.config/lexsi/gate-patterns}"
npat=0
[ -r "$pat" ] && npat=$(grep -c -v -E '^[[:space:]]*(#|$)' "$pat")
if [ ! -r "$pat" ]; then
  bad "pattern file $pat missing or unreadable: identity check NOT run"
elif [ "$npat" -eq 0 ]; then
  bad "pattern file $pat has 0 patterns: identity check NOT run"
else
  clean=$(mktemp); grep -v -E '^[[:space:]]*(#|$)' "$pat" > "$clean"
  git grep -q -i -E -f "$clean" HEAD && bad "identity or referral string at HEAD ($npat patterns)" || ok "no identity or referral strings at HEAD ($npat patterns)"
  # History, not just HEAD: every blob reachable from any ref, every commit and tag message.
  hist=$(git grep -l -i -E -f "$clean" $(git rev-list --all) 2>/dev/null | wc -l | tr -d ' ')
  msgs=$( { git log --all --format='%B'; git for-each-ref --format='%(contents)' refs/tags; } | grep -c -i -E -f "$clean")
  [ "$hist" -eq 0 ] && ok "no identity strings in any blob of any commit" || bad "$hist commit:file hits for identity strings in history"
  [ "$msgs" -eq 0 ] && ok "no identity strings in commit or tag messages" || bad "$msgs commit/tag message lines with identity strings"
  rm -f "$clean"
fi

# 5b. Referral code (patterns in ~/.config/lexsi/gate-referral or GATE_REFERRAL): must be absent at HEAD.
#     Its presence in older history is reported, not failed, until kazani decides keep vs rewrite (HANDOFF).
ref="${GATE_REFERRAL:-$HOME/.config/lexsi/gate-referral}"
nref=0; [ -r "$ref" ] && nref=$(grep -c -v -E '^[[:space:]]*(#|$)' "$ref")
if [ "$nref" -eq 0 ]; then
  bad "referral pattern file $ref missing or empty: referral check NOT run"
else
  rclean=$(mktemp); grep -v -E '^[[:space:]]*(#|$)' "$ref" > "$rclean"
  git grep -q -i -E -f "$rclean" HEAD && bad "referral code at HEAD" || ok "referral code not at HEAD"
  rh=$(for c in $(git rev-list --all); do git grep -q -i -E -f "$rclean" "$c" && echo x; done | wc -l | tr -d ' ')
  echo "INFO  referral code present in $rh of $(git rev-list --all | wc -l | tr -d ' ') commits (history decision: HANDOFF)" | tee -a "$log"
  rm -f "$rclean"
fi

# 6. Every commit authored with the noreply identity
bad_auth=$(git log --all --format='%ae %ce' | grep -v -c "^107648383+kazani-351@users.noreply.github.com 107648383+kazani-351@users.noreply.github.com$")
[ "$bad_auth" -eq 0 ] && ok "all commits use the noreply author/committer email" || bad "$bad_auth commits use another email"

# 7. GitHub's recommended contact file for dual-use research content
[ -f SECURITY.md ] && ok "SECURITY.md present" || bad "SECURITY.md missing (GitHub recommends it for dual-use research repos)"

# 8. Headline numbers rebuild from committed files
python3 -I analyze_labels.py 2>/dev/null | grep -q "plainsft         14/16" && ok "analyze_labels.py rebuilds the 14/16 headline" || bad "analyze_labels.py does not rebuild the headline"

# MANUAL items (kazani confirms; record the decision in HANDOFF.md)
echo "MANUAL  option 1 confirmed at 37 harmful request strings (README disclosure)" | tee -a "$log"
echo "MANUAL  CuratorKIT post-training audit run and reported (rows in/out, no absolute paths in its outputs)" | tee -a "$log"
echo "MANUAL  referral code in history: keep or rewrite decided (see INFO line above)" | tee -a "$log"
[ "$fail" -eq 0 ] && echo "GATE: automatic checks pass; confirm the MANUAL items before pushing" | tee -a "$log" || echo "GATE: BLOCKED" | tee -a "$log"
exit "$fail"
