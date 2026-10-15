"""IE4483 NLP prototype. Run `python sentiment.py --help` for commands."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_table(path):
    """Read CSV, JSON records/columns/index/split, or JSON Lines without guessing labels."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        value = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        raise ValueError("Use a .csv, .json, .jsonl, or .ndjson file.")
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        if {"columns", "data"} <= value.keys():
            return pd.DataFrame(value["data"], columns=value["columns"])
        for wrapper in ("data", "records"):
            if isinstance(value.get(wrapper), list):
                return pd.DataFrame(value[wrapper])
        if value and all(isinstance(v, dict) for v in value.values()):
            first = next(iter(value.values()))
            text_keys = {"reviews", "review", "text", "sentence"}
            if text_keys.intersection(first) and not text_keys.intersection(value):
                return pd.DataFrame.from_dict(value, orient="index").reset_index(names="source_index")
        return pd.DataFrame(value)
    raise ValueError("JSON must contain review records or a table of columns.")


def choose_column(frame, explicit, candidates, required=True):
    if explicit:
        if explicit not in frame:
            raise ValueError(f"Column {explicit!r} not found; available columns: {list(frame.columns)}")
        return explicit
    matches = [name for name in candidates if name in frame]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous columns {matches}; specify the column explicitly.")
    if matches:
        return matches[0]
    if required:
        raise ValueError(f"Expected one of {candidates}; available columns: {list(frame.columns)}")
    return None


def normalise_text(text):
    # Keep negation and punctuation. Normalise whitespace for duplicate detection.
    return re.sub(r"\s+", " ", text).strip()


def load_data(path, labelled, text_col=None, label_col=None, id_col=None):
    frame = read_table(path)
    if frame.empty:
        raise ValueError("The input dataset is empty.")
    text_key = choose_column(frame, text_col, ["reviews", "review", "text", "sentence"])
    id_key = choose_column(frame, id_col, ["id", "ID", "review_id"], required=False)
    raw = frame[text_key]
    if not raw.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Every review must be a non-empty string. Fix missing/empty reviews first.")
    data = pd.DataFrame({"source_row": np.arange(len(frame)), "text": raw.map(normalise_text)})
    if id_key:
        if frame[id_key].isna().any():
            raise ValueError("IDs must not be missing.")
        ids = frame[id_key].astype(str)
        if ids.str.strip().eq("").any() or ids.duplicated().any():
            raise ValueError("IDs must be non-empty and unique.")
        data["id"] = ids
    if labelled:
        label_key = choose_column(frame, label_col, ["sentiments", "sentiment", "label", "labels"])
        labels = pd.to_numeric(frame[label_key], errors="coerce")
        if not labels.isin([0, 1]).all():
            raise ValueError("Labels must be 0 (negative) or 1 (positive); no missing labels are allowed.")
        data["label"] = labels.astype(int)
    # Test labels, if any, are intentionally never used by predict.
    return data


def prepare_training(data, val_size, seed):
    if not 0 < val_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    data = data.copy()
    data["duplicate_key"] = data.text.str.lower()
    conflicts = data.groupby("duplicate_key").label.nunique()
    if (conflicts > 1).any():
        raise ValueError("Identical normalised reviews have conflicting labels. Resolve these before training.")
    original_count = len(data)
    data = data.drop_duplicates("duplicate_key").reset_index(drop=True)
    counts = data.label.value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 2:
        raise ValueError("Need both classes and at least two unique reviews per class.")
    val_count = math.ceil(len(data) * val_size)
    if val_count < 2 or len(data) - val_count < 2:
        raise ValueError("Not enough examples for this stratified split; use more data or adjust validation size.")
    train_idx, val_idx = train_test_split(
        np.arange(len(data)), test_size=val_size, random_state=seed, stratify=data.label
    )
    return data, train_idx, val_idx, original_count - len(data)


def build_model(kind, seed, max_features, ngram_max, c, hidden, alpha, learning_rate):
    if kind == "logistic":
        classifier = LogisticRegression(C=c, max_iter=2000, random_state=seed)
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
            alpha=alpha, learning_rate_init=learning_rate, max_iter=250,
            random_state=seed, early_stopping=False,
        )
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, ngram_max), max_features=max_features,
            min_df=1, sublinear_tf=True, stop_words=None,
        )),
        ("classifier", classifier),
    ])


def fit_with_log(model, x, y):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return time.perf_counter() - started, [str(w.message) for w in captured]


def train(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder is not empty. Pick a new --output to preserve previous experiments.")
    if args.max_features < 1 or args.hidden_units < 1 or args.c <= 0 or args.alpha < 0 or args.learning_rate <= 0:
        raise ValueError("Check feature count, hidden units, C, alpha and learning rate; values are invalid.")
    raw = load_data(args.train, True, args.text_column, args.label_column, args.id_column)
    data, train_idx, val_idx, duplicates = prepare_training(raw, args.validation_size, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    training, validation = data.iloc[train_idx], data.iloc[val_idx]
    split = data[["source_row", "text", "label"]].copy()
    split["split"] = "train"
    split.loc[val_idx, "split"] = "validation"
    split.to_csv(output / "split.csv", index=False)
    models = ["logistic", "mlp"] if args.model == "both" else [args.model]
    results = []
    for kind in models:
        print(f"Training {kind} on {len(training)} reviews; validating on {len(validation)}...", flush=True)
        model = build_model(kind, args.seed, args.max_features, args.ngram_max, args.c,
                            args.hidden_units, args.alpha, args.learning_rate)
        seconds, notes = fit_with_log(model, training.text, training.label)
        predicted = model.predict(validation.text)
        probability = model.predict_proba(validation.text)[:, 1]
        metrics = {
            "model": kind, "validation_accuracy": float(accuracy_score(validation.label, predicted)),
            "validation_macro_f1": float(f1_score(validation.label, predicted, average="macro", zero_division=0)),
            "training_seconds": seconds, "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
            "confusion_matrix_labels": [0, 1],
            "confusion_matrix_rows_true_columns_predicted": confusion_matrix(validation.label, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(validation.label, predicted, labels=[0, 1],
                target_names=["negative", "positive"], output_dict=True, zero_division=0),
            "warnings": notes,
        }
        errors = validation.drop(columns="duplicate_key").copy()
        errors["predicted_label"] = predicted
        errors["probability_positive"] = probability
        errors["correct"] = errors.label == predicted
        errors.to_csv(output / f"{kind}_validation_predictions.csv", index=False)
        write_json(output / f"{kind}_metrics.json", metrics)
        joblib.dump(model, output / f"{kind}_validation_model.joblib")
        results.append(metrics)
        print(f"  accuracy={metrics['validation_accuracy']:.3f}, macro-F1={metrics['validation_macro_f1']:.3f}", flush=True)

    summary_cols = ["model", "validation_accuracy", "validation_macro_f1", "training_seconds", "vocabulary_size"]
    pd.DataFrame(results)[summary_cols].to_csv(output / "comparison.csv", index=False)
    best = max(results, key=lambda r: (r["validation_macro_f1"], r["validation_accuracy"]))
    print(f"Refitting selected model ({best['model']}) on all {len(data)} unique training reviews...", flush=True)
    final = build_model(best["model"], args.seed, args.max_features, args.ngram_max, args.c,
                        args.hidden_units, args.alpha, args.learning_rate)
    refit_seconds, refit_warnings = fit_with_log(final, data.text, data.label)
    artifact = {
        "pipeline": final, "selected_model": best["model"], "labels": {0: "negative", 1: "positive"},
        "training_text_hashes": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in data.duplicate_key],
    }
    joblib.dump(artifact, output / "model.joblib")
    configuration = vars(args).copy()
    configuration.pop("handler", None)
    write_json(output / "run.json", {
        "configuration": configuration, "training_file_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]},
        "input_rows": len(raw), "duplicate_reviews_removed": duplicates,
        "train_rows": len(training), "validation_rows": len(validation),
        "label_counts_after_deduplication": {str(k): int(v) for k, v in data.label.value_counts().items()},
        "median_words": float(data.text.str.split().str.len().median()),
        "selected_model": best["model"], "selection_metric": "validation_macro_f1",
        "final_refit_seconds": refit_seconds, "final_refit_warnings": refit_warnings,
        "note": "Validation metrics belong to split-trained models; final model is refit on all labelled input. Synthetic demo metrics are not assignment results.",
    })
    print(f"Saved {output / 'model.joblib'} and validation evidence.")


def load_artifact(path):
    # Joblib uses pickle: load only models generated by you or a trusted source.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Use the final model.joblib produced by the train command.")
    return artifact


def save_predictions(data, predicted, output, template=None, allow_generated_ids=False):
    if "id" not in data:
        if not allow_generated_ids:
            raise ValueError("Test data has no IDs. Provide --id-column or explicitly use --allow-generated-ids only if row-number IDs are confirmed by the course. A template alone cannot establish row alignment.")
        data = data.copy()
        data["id"] = np.arange(len(data)).astype(str)
    result = pd.DataFrame({"id": data.id.to_numpy(), "label": np.asarray(predicted, dtype=int)})
    if template:
        supplied = pd.read_csv(template, dtype=str, keep_default_na=False)
        if list(supplied.columns) != ["id", "label"]:
            raise ValueError("Submission template must have exactly the columns id,label.")
        if supplied.id.duplicated().any() or set(supplied.id) != set(result.id):
            raise ValueError("Submission template IDs do not match the test IDs exactly.")
        result = supplied[["id"]].merge(result, on="id", how="left", validate="one_to_one", sort=False)
    output = Path(output)
    if output.exists():
        raise ValueError("Prediction output already exists. Choose a new --output to preserve it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def predict(args):
    artifact = load_artifact(args.model_file)
    data = load_data(args.test, False, args.text_column, id_col=args.id_column)
    hashes = {hashlib.sha256(t.lower().encode("utf-8")).hexdigest() for t in data.text}
    overlap = hashes.intersection(artifact.get("training_text_hashes", []))
    if overlap:
        warnings.warn(f"{len(overlap)} unique test texts also appear in training; check for overlap before interpreting results.")
    predicted = artifact["pipeline"].predict(data.text)
    result = save_predictions(data, predicted, args.output, args.template, args.allow_generated_ids)
    print(f"Wrote {len(result)} predictions to {args.output}; 0=negative, 1=positive.")


def classify_text(model_path, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty sentence or review.")
    artifact = load_artifact(model_path)
    model = artifact["pipeline"]
    review = normalise_text(text)
    label = int(model.predict([review])[0])
    probability = float(model.predict_proba([review])[0, 1])
    return {"text": text, "label": label, "sentiment": artifact["labels"][label],
            "probability_positive": probability, "model": artifact["selected_model"],
            "note": "Probability is a model estimate, not a calibrated certainty measure."}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Validate models, compare them, then refit the winner.")
    training.add_argument("--train", required=True)
    training.add_argument("--output", default="runs/experiment-01")
    training.add_argument("--model", choices=["logistic", "mlp", "both"], default="both")
    training.add_argument("--text-column")
    training.add_argument("--label-column")
    training.add_argument("--id-column")
    training.add_argument("--validation-size", type=float, default=0.2)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--max-features", type=int, default=10000)
    training.add_argument("--ngram-max", type=int, choices=[1, 2], default=2)
    training.add_argument("--c", type=float, default=1.0)
    training.add_argument("--hidden-units", type=int, default=64)
    training.add_argument("--alpha", type=float, default=0.001)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.set_defaults(handler=train)
    prediction = commands.add_parser("predict", help="Export id,label predictions without using test labels.")
    prediction.add_argument("--model-file", required=True)
    prediction.add_argument("--test", required=True)
    prediction.add_argument("--output", default="runs/submission.csv")
    prediction.add_argument("--template")
    prediction.add_argument("--text-column")
    prediction.add_argument("--id-column")
    prediction.add_argument("--allow-generated-ids", action="store_true")
    prediction.set_defaults(handler=predict)
    sentence = commands.add_parser("sentence", help="Classify a sentence or complete review.")
    sentence.add_argument("--model-file", required=True)
    sentence.add_argument("--text", required=True)
    sentence.set_defaults(handler=lambda a: print(json.dumps(classify_text(a.model_file, a.text), indent=2)))
    return root


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
