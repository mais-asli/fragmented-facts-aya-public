"""Check the actual Aya repository and small config file; never print credentials."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fragmented_facts.io import now, write_json


def main():
    from huggingface_hub import HfApi, hf_hub_download
    report = {"model_id": "CohereLabs/aya-23-8B", "checked_at": now(), "weights_downloaded": False,
              "scope": "Repository access and small model configuration only"}
    try:
        info = HfApi().model_info(report["model_id"])
        report["revision"] = info.sha
        report["gated"] = info.gated
        path = hf_hub_download(report["model_id"], "config.json", revision=info.sha,
                               local_dir="data/raw/aya_configuration")
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        report.update(status="configuration_accessible",
            architecture=config.get("model_type"), layers=config.get("num_hidden_layers"),
            context_length=config.get("max_position_embeddings"))
    except Exception as exc:
        response = getattr(exc, "response", None)
        report.update(status="access_check_failed", error_type=type(exc).__name__,
                      http_status=response.status_code if response is not None else None)
    write_json("results/aya_access_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
