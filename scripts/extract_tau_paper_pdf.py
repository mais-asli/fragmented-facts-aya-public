"""Verify and save the PDF returned by the private TAU compilation workflow."""

import base64
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "results/tau_paper_compile_workflow.json"
PREVIEW_DIR = ROOT / "tmp/pdfs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tmp/pdfs/Fragmented_Facts_Aya_preview.pdf")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(PREVIEW_DIR.resolve()) or output.suffix.lower() != ".pdf":
        raise ValueError("Preview output must be a PDF under tmp/pdfs")
    state = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    event = state.get("workflow", {})
    if (state.get("status") != "workflow_finished" or
            state.get("workflow_exit_code") != 0 or
            event.get("status") != "paper_compilation_finished"):
        raise ValueError("TAU paper compilation has not succeeded")
    data = base64.b64decode(event["pdf_base64"], validate=True)
    if (len(data) != event["pdf_bytes"] or
            hashlib.sha256(data).hexdigest() != event["pdf_sha256"] or
            not data.startswith(b"%PDF-")):
        raise ValueError("Compiled PDF checksum or format mismatch")
    if output.exists():
        raise FileExistsError("Preserve existing paper PDF; inspect before replacing")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(json.dumps({"output": str(output), "bytes": len(data),
                      "sha256": event["pdf_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
