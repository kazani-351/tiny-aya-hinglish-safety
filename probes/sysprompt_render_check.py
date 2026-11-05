# Nov 3 probe 12 (tokenizer only, CPU): does a user-supplied system message REPLACE Tiny Aya's built-in preamble
# or sit alongside it? Side-agent question 2026-11-03: if it replaces it, probe 11's base_sys gain could come from
# deleting the ~366-token default block (which lists training languages incl. Zulu), not from our instruction.
# Run: ./run_cpu.sh probes/sysprompt_render_check.py lexsi-sysrender   (~3 min; no model weights downloaded)
import os, subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], capture_output=True, text=True)
from huggingface_hub import login
def _hf_login(path="/content/hf.env"):
    token = None
    with open(path) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    os.remove(path)
    assert token
    login(token=token)
_hf_login()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
SYS = ("Tum ek madadgaar assistant ho. Jab user Hinglish mein likhe (Roman script mein Hindi aur English ka mix), "
       "to hamesha Hinglish mein hi jawab do, Roman script mein. English, Devanagari ya kisi aur bhasha mein mat badlo.")
plain = tok.apply_chat_template([{"role": "user", "content": "test"}], add_generation_prompt=True, tokenize=False)
withsys = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": "test"}],
                                  add_generation_prompt=True, tokenize=False)
cut = plain.index("test")
default_block = plain[:cut]
print(f"plain render: {len(tok(plain)['input_ids'])} tokens | with system: {len(tok(withsys)['input_ids'])} tokens")
print("Zulu in plain:", "Zulu" in plain, "| Zulu in with-system:", "Zulu" in withsys)
# Longest chunk of the default block that survives verbatim in the with-system render.
lines = [l for l in default_block.splitlines() if len(l.strip()) > 20]
kept = [l for l in lines if l in withsys]
print(f"default-block lines (>20 chars) kept verbatim: {len(kept)}/{len(lines)}")
print("VERDICT:", "built-in block SURVIVES alongside our system message" if len(kept) == len(lines)
      else "built-in block is REPLACED or altered by our system message")
print("--- plain render, up to user turn ---\n" + default_block[-1500:])
print("--- with-system render, up to user turn ---\n" + withsys[:withsys.index('test')][-1500:])
print("RUN_OK")
