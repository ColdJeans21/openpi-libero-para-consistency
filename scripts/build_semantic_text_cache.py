"""Build training-only frozen T5 features. Evaluation text is never read here."""

import argparse
import json
import pathlib

import numpy as np

from openpi.shared.semantic_text import FrozenT5Encoder
from openpi.shared.semantic_text import canonical_text
from openpi.shared.semantic_text import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paraphrases", default="examples/libero_para/train_paraphrases_v1.json")
    parser.add_argument("--model-dir", default="/data/openpi_models/t5-large")
    parser.add_argument("--output", default="assets/semantic_interface/t5_large_train_v1.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen feature asset: {output}")
    source = json.loads(pathlib.Path(args.paraphrases).read_text())["paraphrases"]
    prompts = sorted({canonical_text(text) for key, values in source.items() for text in [key, *values]})
    model_dir = pathlib.Path(args.model_dir).resolve()
    encoder = FrozenT5Encoder(str(model_dir), device=args.device)
    features = np.concatenate(
        [encoder.encode(prompts[start : start + args.batch_size]) for start in range(0, len(prompts), args.batch_size)]
    )
    if features.shape != (len(prompts), 1024) or not np.isfinite(features).all():
        raise ValueError(f"Unexpected T5-large features: {features.shape}")
    metadata = {
        "schema_version": 1,
        "model_id": "google-t5/t5-large",
        "model_dir": str(model_dir),
        "max_length": encoder.max_length,
        "pooling": "attention-mask mean of final contextual encoder states; float32; eval mode",
        "encoder_files_sha256": {
            name: file_sha256(model_dir / name)
            for name in ("config.json", "model.safetensors", "spiece.model", "tokenizer.json")
        },
        "training_source": args.paraphrases,
        "training_source_sha256": file_sha256(args.paraphrases),
        "prompt_count": len(prompts),
        "evaluation_text_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, texts=np.asarray(prompts), features=features, metadata=json.dumps(metadata))
    print(json.dumps({**metadata, "cache_sha256": file_sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
