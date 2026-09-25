"""Reproduce the paper's lexical, embedding, ordinal, and length diagnostics."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REFERENCE_EMBEDDINGS = ROOT / "data/modernbert_embeddings.npz"
DATA_SHA256 = "112395b9fe0bffb8ac09f95dd180bb64ba82bf7a5c0f34df152f20f65fdbd5e4"
REFERENCE_EMBEDDING_SHA256 = "dad371fd76637b14e93ebc054b0ca0f2acc2b6c169ee6f2cbd9e95f7843d2024"
MODEL = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
LEVELS = (-2, -1, 0, 1, 2)
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SECOND_PERSON_RE = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b", re.I)
SCHEMES = {
    "three_class": {-2: "cold", -1: "cold", 0: "neutral", 1: "warm", 2: "warm"},
    "five_class": {-2: "very_cold", -1: "cold", 0: "neutral", 1: "warm", 2: "very_warm"},
}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_data(path):
    if sha_file(path) != DATA_SHA256:
        raise ValueError("Dataset differs from the version evaluated in the paper")
    families = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    records = []
    for index, family in enumerate(families):
        # Preserve these literal strings: hashing groups changes their sorted
        # order and therefore can change seeded StratifiedGroupKFold splits.
        group = (f"legacy::{family['prompt_id']}" if index < 300 else
                 json.dumps([family.get("domain"), tuple(family["spec"].get("key_facts", [])),
                             family["spec"].get("invariant_answer")], ensure_ascii=False, sort_keys=True))
        if [r["warmth_level"] for r in family["responses"]] != list(LEVELS):
            raise ValueError("Expected exactly five responses in numeric-label order")
        for response in family["responses"]:
            records.append({"id": f"{family['prompt_id']}::{response['warmth_level']}",
                            "family": family["prompt_id"], "group": group,
                            "label": response["warmth_level"], "text": response["text"]})
    if len(families) != 1000 or len({r["id"] for r in records}) != 5000 or len({r["group"] for r in records}) != 650:
        raise ValueError("Unexpected dataset structure")
    return records


def splits_for(records, target):
    groups = np.asarray([r["group"] for r in records])
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                  .split(np.zeros((len(records), 1)), target, groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Factual-core leakage")
    return splits


def surface_matrix(records):
    rows = []
    for record in records:
        text = record["text"]
        words = WORD_RE.findall(text)
        rows.append([len(words), len(text), len(SENTENCE_RE.split(text)), text.count("?"), text.count("!"),
                     len(SECOND_PERSON_RE.findall(text)), sum(w.lower() in {"we", "our", "us"} for w in words)])
    return np.asarray(rows, dtype=np.float64)


def classify(features, target, splits, *, tfidf=False):
    predicted = np.empty(len(target), dtype=object)
    for fold, (train, test) in enumerate(splits, 1):
        if tfidf:
            model = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                                 max_features=60_000, sublinear_tf=True),
                                  LogisticRegression(C=4.0, max_iter=3000, random_state=42 + fold))
            x_train, x_test = features[train].tolist(), features[test].tolist()
        else:
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=5000, random_state=42 + fold))
            x_train, x_test = features[train], features[test]
        model.fit(x_train, target[train])
        predicted[test] = model.predict(x_test)
    return predicted.astype(str), {"accuracy": float(accuracy_score(target, predicted)), "responses": len(target)}


def ordinal_scores(features, target, splits):
    scores = np.full(len(target), np.nan, dtype=np.float64)
    for train, test in splits:
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(features[train], target[train])
        scores[test] = model.predict(features[test])
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete/nonfinite ordinal predictions")
    return scores


def ordering(records, scores):
    families = {}
    for row, score in zip(records, scores):
        families.setdefault(row["family"], {})[row["label"]] = float(score)
    all_correct = adjacent = cross = strict = 0
    for values in families.values():
        ordered = [values[label] for label in LEVELS]
        all_correct += sum(ordered[i] < ordered[j] for i in range(5) for j in range(i + 1, 5))
        adjacent += sum(ordered[i] < ordered[i + 1] for i in range(4))
        cross += sum(values[cold] < values[warm] for cold in (-2, -1) for warm in (1, 2))
        strict += all(ordered[i] < ordered[i + 1] for i in range(4))
    count = len(families)
    return {"families": count, "all_pair_accuracy": all_correct / (10 * count),
            "adjacent_accuracy": adjacent / (4 * count), "cross_polarity_accuracy": cross / (4 * count),
            "strictly_ordered_family_rate": strict / count}


def length_metrics(records):
    counts = surface_matrix(records)[:, 0].reshape(-1, 5)
    delta = counts[:, 4] - counts[:, 0]
    rho = [float(spearmanr(LEVELS, row).statistic) for row in counts if np.ptp(row) != 0]
    return {"families": len(counts), "plus2_longer": int((delta > 0).sum()),
            "plus2_longer_rate": float((delta > 0).mean()),
            "mean_plus2_minus_minus2_words": float(delta.mean()),
            "mean_within_family_spearman": float(np.mean(rho)),
            "undefined_spearman_families": len(counts) - len(rho)}


def extract_embeddings(records, *, device="auto", local_files_only=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("The reference CUDA extraction requires BF16 support; use --device cpu otherwise")
    precision = "bf16" if selected == "cuda" else "fp32"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=local_files_only,
                                               trust_remote_code=False)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, local_files_only=local_files_only,
                                     trust_remote_code=False, attn_implementation="sdpa")
    texts = [r["text"] for r in records]
    lengths = [len(ids) for ids in tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)["input_ids"]]
    if max(lengths) > int(model.config.max_position_embeddings):
        raise ValueError("Refusing to truncate a response")
    model.to(selected).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            encoded = tokenizer(texts[start:start + 8], padding=True, truncation=False, add_special_tokens=True,
                                return_special_tokens_mask=True, return_tensors="pt")
            special = encoded.pop("special_tokens_mask").bool().to(selected)
            inputs = {key: value.to(selected) for key, value in encoded.items()}
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()
            with context:
                hidden = model(**inputs).last_hidden_state
            valid = inputs["attention_mask"].bool() & ~special
            empty = valid.sum(1) == 0
            if empty.any():
                valid[empty] = inputs["attention_mask"].bool()[empty]
            weights = valid.unsqueeze(-1).to(hidden.dtype)
            batches.append(((hidden * weights).sum(1) / weights.sum(1).clamp_min(1)).float().cpu().numpy())
            if start % 400 == 0 or start + 8 >= len(texts):
                print(f"Embedded {min(start + 8, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(batches).astype(np.float32, copy=False), {
        "model": MODEL, "revision": REVISION, "precision": precision, "device": selected,
        "pooling": "final-layer mean over non-special answer tokens", "longest_input_tokens": max(lengths),
        "truncated_responses": 0, "batch_size": 8}


def get_embeddings(records, output, args):
    fresh = getattr(args, "fresh_embeddings", False)
    path = args.embeddings or (output / "embeddings.npz" if fresh else REFERENCE_EMBEDDINGS)
    if fresh and path.exists():
        raise FileExistsError("Fresh extraction requires a new embedding output path")
    manifest = path.with_suffix(".json")
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            matrix = archive["embeddings"]
        fingerprint = sha_array(matrix)
        if fingerprint == REFERENCE_EMBEDDING_SHA256:
            metadata = {"model": MODEL, "revision": REVISION, "precision": "bf16",
                        "verified_reference_embedding": True, "array_sha256": fingerprint,
                        "data_sha256": DATA_SHA256}
        elif manifest.exists():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if (metadata["data_sha256"] != DATA_SHA256 or metadata["array_sha256"] != fingerprint or
                    metadata["revision"] != REVISION or metadata["model"] != MODEL or
                    metadata["row_ids"] != [r["id"] for r in records]):
                raise ValueError("Embedding-cache provenance mismatch")
        else:
            raise ValueError("Unrecognized embedding cache without matching provenance")
    else:
        if not fresh:
            raise FileNotFoundError(f"Missing embedding cache: {path}. Use --fresh-embeddings to extract new features.")
        matrix, metadata = extract_embeddings(records, device=args.device, local_files_only=args.local_files_only)
        np.savez_compressed(path, embeddings=matrix)
        metadata.update(data_sha256=DATA_SHA256, array_sha256=sha_array(matrix), row_ids=[r["id"] for r in records])
        save_json(manifest, metadata)
    if matrix.shape != (5000, 768) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("Expected finite float32 5000 x 768 answer embeddings")
    return matrix, metadata


def evaluate(records, embeddings=None, *, include_pca=False):
    target = np.asarray([r["label"] for r in records])
    texts = np.asarray([r["text"] for r in records])
    surfaces = surface_matrix(records)
    metrics = {"dataset": {"families": 1000, "responses": 5000, "factual_cores": 650},
               "classification": {}, "ordinal": {}, "length": length_metrics(records)}
    predictions, split_arrays = {}, {}
    features = {"word_count": surfaces[:, :1], "surface_seven": surfaces}
    if embeddings is not None:
        features["modernbert_full768"] = embeddings
    if embeddings is not None and include_pca:
        # This reproduces the original global, label-free PCA (not fold-local).
        pca = PCA(n_components=128, whiten=False, svd_solver="full")
        projected = pca.fit_transform(np.asarray(embeddings, dtype=np.float64)).astype(np.float32)
        features["modernbert_pca128"] = projected
        metrics["pca_variance"] = {str(d): float(pca.explained_variance_ratio_[:d].sum()) for d in (64, 128)}
    for scheme, mapping in SCHEMES.items():
        labels = np.asarray([mapping[label] for label in target])
        splits = splits_for(records, labels)
        for fold, (train, test) in enumerate(splits):
            split_arrays[f"{scheme}_{fold}_train"] = train
            split_arrays[f"{scheme}_{fold}_test"] = test
        for name, matrix in {"tfidf": texts, **features}.items():
            prediction, result = classify(matrix, labels, splits, tfidf=name == "tfidf")
            predictions[f"{name}_{scheme}"] = prediction
            metrics["classification"].setdefault(name, {})[scheme] = result
            print(f"{name} {scheme}: {result['accuracy']:.4f}", flush=True)
    if embeddings is not None:
        splits = splits_for(records, target)
        for fold, (train, test) in enumerate(splits):
            split_arrays[f"ordinal_{fold}_train"] = train
            split_arrays[f"ordinal_{fold}_test"] = test
        ordinal_features = {"modernbert_full768": embeddings}
        if include_pca:
            ordinal_features["modernbert_pca64"] = projected[:, :64].copy(order="K")
        for name, matrix in ordinal_features.items():
            scores = ordinal_scores(matrix, target, splits)
            predictions[name + "_ordinal"] = scores
            metrics["ordinal"][name] = ordering(records, scores)
    return metrics, predictions, split_arrays


def run(args):
    """Run and save numerical experiments; return their measured results."""
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if any((args.output / name).exists() for name in ("results.json", "predictions.npz", "splits.npz")):
        raise FileExistsError("Results exist; choose a new --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    records = load_data(args.data)
    with threadpool_limits(limits=args.threads), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        embeddings, metadata = (None, None) if args.text_only else get_embeddings(records, args.output, args)
        metrics, predictions, splits = evaluate(records, embeddings, include_pca=args.include_pca)
    metrics["method"] = {"model": MODEL, "revision": REVISION, "data_sha256": DATA_SHA256,
                         "embedding_provenance": {k: v for k, v in (metadata or {}).items() if k != "row_ids"},
                         "cv": "5-fold StratifiedGroupKFold, shuffle=True, random_state=42",
                         "scaler": "fold-local StandardScaler for numeric features only",
                         "ordinal_estimator": "LinearRegression (ordinary least squares)",
                         "pca_fit_scope": ("all responses, without labels; non-whitened full SVD"
                                           if args.include_pca and embeddings is not None else None),
                         "source_sha256": sha_file(Path(__file__))}
    save_json(args.output / "results.json", metrics)
    np.savez_compressed(args.output / "predictions.npz", row_ids=np.asarray([r["id"] for r in records]), **predictions)
    np.savez_compressed(args.output / "splits.npz", **splits)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/warmth_families.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    cache_options = parser.add_mutually_exclusive_group()
    cache_options.add_argument("--embeddings", type=Path, help="Override the bundled reference embedding NPZ")
    cache_options.add_argument("--fresh-embeddings", action="store_true", help="Extract new embeddings instead of using the bundled reference")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--text-only", action="store_true", help="Skip only the embedding-based analyses")
    parser.add_argument("--include-pca", action="store_true", help="Also run the earlier PCA comparison")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    metrics = run(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
