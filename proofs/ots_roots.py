"""Print the Bitcoin block heights and merkle roots an .ots proof commits to.

Use when `ots verify` can't reach a local Bitcoin node. Compare each printed
root with the block's merkle root on any block explorer, for example
https://blockstream.info/api/block-height/<height> then /api/block/<hash>.

    python proofs/ots_roots.py proofs/commit-<sha>.sha.ots proofs/commit-<sha>.sha
"""
import hashlib
import sys

from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
from opentimestamps.core.serialize import StreamDeserializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile

ots_path, target_path = sys.argv[1], sys.argv[2]
with open(ots_path, "rb") as f:
    proof = DetachedTimestampFile.deserialize(StreamDeserializationContext(f))
with open(target_path, "rb") as f:
    if proof.file_digest != hashlib.sha256(f.read()).digest():
        sys.exit("FAIL: the proof does not match this file")
print("file digest matches the proof")
for msg, att in sorted(proof.timestamp.all_attestations(), key=lambda x: getattr(x[1], "height", 0)):
    if isinstance(att, BitcoinBlockHeaderAttestation):
        print(f"block {att.height}  merkle root {msg[::-1].hex()}")
