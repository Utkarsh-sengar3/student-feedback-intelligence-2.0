"""
PHASE 10 — Transformer Sentiment Baseline
EduRABSA Student Feedback Sentiment Intelligence

Add this file to:
    student-feedback-intelligence/scripts/phase10_transformer_sentiment.py

Run from project root:
    python -u scripts\phase10_transformer_sentiment.py

Requirements:
    pip install torch transformers datasets accelerate scikit-learn pandas joblib

Protocol:
- train_phase2.csv is the labeled train_validation source.
- test_phase2.csv remains untouched during model selection.
- 80/20 stratified split of train_validation.
- Model selection uses validation Macro F1.
- The selected transformer is retrained on all labeled train_validation rows.
- Final evaluation is performed once on the 2499 labeled test rows.

Default model:
    distilbert-base-uncased

If the model cannot be downloaded because of network restrictions, the script
will stop with a clear error rather than silently producing an invalid result.
"""

from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "train_phase2.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "test_phase2.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "edurabsa" / "phase10"
MODEL_DIR = PROJECT_ROOT / "models" / "sentiment" / "phase10_transformer"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "text_clean"
LABEL_COL = "overall_sentiment"

LABEL2ID = {
    "Neg": 0,
    "Neu": 1,
    "Pos": 2,
}

ID2LABEL = {
    0: "Neg",
    1: "Neu",
    2: "Pos",
}

CLASSES = ["Neg", "Neu", "Pos"]

RANDOM_STATE = 42

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256

EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01


def calculate_metrics(y_true, y_pred):

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )[2]

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision.mean()
        ),
        "recall_macro": float(
            recall.mean()
        ),
        "f1_macro": float(
            f1.mean()
        ),
        "f1_weighted": float(
            weighted_f1
        ),
    }


class FeedbackDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length,
    ):

        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        item = {
            key: torch.tensor(value)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return item


def main():

    print("=" * 78)
    print("PHASE 10 — TRANSFORMER SENTIMENT BASELINE")
    print("=" * 78)

    # ---------------------------------------------------------------
    # Environment
    # ---------------------------------------------------------------
    print("\nDevice:")
    print(
        "CUDA GPU"
        if torch.cuda.is_available()
        else "CPU"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"\nTransformer: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # ---------------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------------
    if not TRAIN_PATH.exists():

        raise FileNotFoundError(
            f"\nTraining file not found:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"\nTest file not found:\n{TEST_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    required = {
        TEXT_COL,
        LABEL_COL,
    }

    missing_train = (
        required - set(train.columns)
    )

    missing_test = (
        required - set(test.columns)
    )

    if missing_train:

        raise ValueError(
            "Training file missing columns: "
            f"{sorted(missing_train)}"
        )

    if missing_test:

        raise ValueError(
            "Test file missing columns: "
            f"{sorted(missing_test)}"
        )

    # ---------------------------------------------------------------
    # Keep only labeled sentiment rows
    # ---------------------------------------------------------------
    train = train[
        train[LABEL_COL].isin(CLASSES)
    ].copy()

    test = test[
        test[LABEL_COL].isin(CLASSES)
    ].copy()

    train[TEXT_COL] = (
        train[TEXT_COL]
        .fillna("")
        .astype(str)
    )

    test[TEXT_COL] = (
        test[TEXT_COL]
        .fillna("")
        .astype(str)
    )

    train["label_id"] = train[
        LABEL_COL
    ].map(LABEL2ID)

    test["label_id"] = test[
        LABEL_COL
    ].map(LABEL2ID)

    # ---------------------------------------------------------------
    # Fixed validation split
    # ---------------------------------------------------------------
    train_part, val_part = train_test_split(
        train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train[LABEL_COL],
    )

    print(
        f"\nLabeled train_validation rows: "
        f"{len(train)}"
    )

    print(
        f"Internal training rows: "
        f"{len(train_part)}"
    )

    print(
        f"Internal validation rows: "
        f"{len(val_part)}"
    )

    print(
        f"Labeled final test rows: "
        f"{len(test)}"
    )

    # ---------------------------------------------------------------
    # Seed
    # ---------------------------------------------------------------
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    set_seed(RANDOM_STATE)

    # ---------------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------------
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------
    train_dataset = FeedbackDataset(
        train_part[TEXT_COL].values,
        train_part["label_id"].values,
        tokenizer,
        MAX_LENGTH,
    )

    validation_dataset = FeedbackDataset(
        val_part[TEXT_COL].values,
        val_part["label_id"].values,
        tokenizer,
        MAX_LENGTH,
    )

    test_dataset = FeedbackDataset(
        test[TEXT_COL].values,
        test["label_id"].values,
        tokenizer,
        MAX_LENGTH,
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    print(
        "\nLoading pretrained transformer..."
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ---------------------------------------------------------------
    # Validation metric
    # ---------------------------------------------------------------
    def compute_metrics(eval_prediction):

        predictions, labels = eval_prediction

        predictions = np.argmax(
            predictions,
            axis=-1,
        )

        return calculate_metrics(
            labels,
            predictions,
        )

    # ---------------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(
            OUTPUT_DIR / "training_runs"
        ),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=25,
        report_to="none",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TRANSFORMER TRAINING")
    print("=" * 78)

    trainer.train()

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BEST VALIDATION MODEL")
    print("=" * 78)

    validation_result = trainer.evaluate(
        validation_dataset
    )

    print(validation_result)

    validation_macro_f1 = float(
        validation_result[
            "eval_f1_macro"
        ]
    )

    # Save best validation model.
    trainer.save_model(
        str(MODEL_DIR)
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR)
    )

    # ---------------------------------------------------------------
    # Final test
    #
    # IMPORTANT:
    # Test is evaluated only after validation-based model selection.
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("FINAL TEST EVALUATION")
    print("=" * 78)

    test_result = trainer.predict(
        test_dataset
    )

    test_logits = test_result.predictions

    test_pred_ids = np.argmax(
        test_logits,
        axis=-1,
    )

    test_true_ids = test[
        "label_id"
    ].values

    final_metrics = calculate_metrics(
        test_true_ids,
        test_pred_ids,
    )

    print(
        f"Accuracy       : "
        f"{final_metrics['accuracy']:.6f}"
    )

    print(
        f"Macro Precision: "
        f"{final_metrics['precision_macro']:.6f}"
    )

    print(
        f"Macro Recall   : "
        f"{final_metrics['recall_macro']:.6f}"
    )

    print(
        f"Macro F1       : "
        f"{final_metrics['f1_macro']:.6f}"
    )

    print(
        f"Weighted F1    : "
        f"{final_metrics['f1_weighted']:.6f}"
    )

    # ---------------------------------------------------------------
    # Comparison against classical champion
    # ---------------------------------------------------------------
    phase9_3_macro_f1 = 0.7049

    improvement = (
        final_metrics["f1_macro"]
        - phase9_3_macro_f1
    )

    print(
        "\nChange vs Phase 9.3: "
        f"{improvement:+.6f}"
    )

    if improvement > 0:
        verdict = (
            "Transformer beats the Phase 9.3 "
            "classical benchmark."
        )
    else:
        verdict = (
            "Transformer does not beat the "
            "Phase 9.3 classical benchmark."
        )

    print(
        f"\nVERDICT: {verdict}"
    )

    # ---------------------------------------------------------------
    # Test artifacts
    # ---------------------------------------------------------------
    cm = confusion_matrix(
        test_true_ids,
        test_pred_ids,
        labels=[0, 1, 2],
    )

    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(
        OUTPUT_DIR /
        "final_test_confusion_matrix.csv"
    )

    report = classification_report(
        test_true_ids,
        test_pred_ids,
        labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        OUTPUT_DIR /
        "final_test_classification_report.csv"
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------
    prediction_columns = [
        c
        for c in [
            "review_id",
            "original_id",
            "domain",
            "text",
            "text_clean",
            "overall_sentiment",
        ]
        if c in test.columns
    ]

    prediction_output = test[
        prediction_columns
    ].copy()

    prediction_output[
        "predicted_sentiment"
    ] = [
        ID2LABEL[int(x)]
        for x in test_pred_ids
    ]

    prediction_output.to_csv(
        OUTPUT_DIR /
        "final_test_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------------
    summary = {
        "phase": "10",
        "status": "completed",
        "model": MODEL_NAME,
        "protocol": {
            "labeled_train_validation_rows": int(
                len(train)
            ),
            "internal_train_rows": int(
                len(train_part)
            ),
            "internal_validation_rows": int(
                len(val_part)
            ),
            "labeled_test_rows_evaluated": int(
                len(test)
            ),
            "selection_metric": (
                "validation_macro_f1"
            ),
            "test_used_for_model_selection": False,
        },
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "weight_decay": WEIGHT_DECAY,
            "random_state": RANDOM_STATE,
            "device": (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),
        },
        "validation": {
            "macro_f1": validation_macro_f1,
            "raw_trainer_metrics": {
                str(k): (
                    float(v)
                    if isinstance(
                        v,
                        (float, int),
                    )
                    else v
                )
                for k, v in validation_result.items()
            },
        },
        "final_test": final_metrics,
        "comparison": {
            "phase9_3_macro_f1": phase9_3_macro_f1,
            "phase10_macro_f1": final_metrics[
                "f1_macro"
            ],
            "macro_f1_change": improvement,
            "beats_phase9_3": improvement > 0,
            "verdict": verdict,
        },
        "model_path": str(
            MODEL_DIR
        ),
    }

    summary_path = (
        OUTPUT_DIR /
        "phase10_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # README
    # ---------------------------------------------------------------
    readme = f"""# Phase 10 — Transformer Sentiment Baseline

## Model

`{MODEL_NAME}`

## Dataset protocol

- Labeled train-validation: {len(train)}
- Internal train: {len(train_part)}
- Internal validation: {len(val_part)}
- Labeled final test: {len(test)}

The test set was not used for model selection.

## Training

- Epochs: {EPOCHS}
- Learning rate: {LEARNING_RATE}
- Train batch size: {TRAIN_BATCH_SIZE}
- Evaluation batch size: {EVAL_BATCH_SIZE}
- Maximum sequence length: {MAX_LENGTH}
- Weight decay: {WEIGHT_DECAY}

## Validation Macro F1

`{validation_macro_f1:.6f}`

## Final Test

Accuracy:

`{final_metrics["accuracy"]:.6f}`

Macro F1:

`{final_metrics["f1_macro"]:.6f}`

Weighted F1:

`{final_metrics["f1_weighted"]:.6f}`

## Classical champion

Phase 9.3 Macro F1:

`{phase9_3_macro_f1:.6f}`

Transformer change:

`{improvement:+.6f}`

## Verdict

{verdict}

## Artifacts

- `phase10_summary.json`
- `final_test_confusion_matrix.csv`
- `final_test_classification_report.csv`
- `final_test_predictions.csv`
- `models/sentiment/phase10_transformer/`
"""

    (OUTPUT_DIR / "PHASE_10_README.md").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("PHASE 10 COMPLETE")
    print("=" * 78)
    print(
        f"Transformer: {MODEL_NAME}"
    )
    print(
        f"Validation Macro F1: "
        f"{validation_macro_f1:.4f}"
    )
    print(
        f"Final Test Macro F1: "
        f"{final_metrics['f1_macro']:.4f}"
    )
    print(
        f"Change vs Phase 9.3: "
        f"{improvement:+.4f}"
    )

    print(
        "\nSummary:"
    )
    print(summary_path)

    print(
        "\nModel:"
    )
    print(MODEL_DIR)


if __name__ == "__main__":
    main()
