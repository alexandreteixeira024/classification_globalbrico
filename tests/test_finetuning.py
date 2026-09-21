import tempfile
import unittest
from pathlib import Path

from src.finetuning import assign_stable_folds, paired_with_previous, save_json, write_csv


LABELS = ("Pedido de Informação", "Pedido de Encomenda", "SPAM")


def examples_per_class(count=5):
    examples = []
    for label_index, label in enumerate(LABELS):
        for item_index in range(count):
            uid = f"{label_index}-{item_index}"
            examples.append((uid, f"texto {uid}", label, f"subject:{uid}"))
    return examples


class CrossValidationFoldTests(unittest.TestCase):
    def test_each_example_is_in_exactly_one_balanced_fold(self):
        examples = examples_per_class()
        with tempfile.TemporaryDirectory() as directory:
            folds, _ = assign_stable_folds(
                examples, Path(directory) / "folds.json", folds=5, seed=42
            )

        observed = [item[0] for fold in folds for item in fold]
        self.assertCountEqual(observed, [item[0] for item in examples])
        self.assertEqual(len(observed), len(set(observed)))
        for fold in folds:
            self.assertCountEqual([item[2] for item in fold], LABELS)

    def test_existing_groups_keep_their_fold_when_data_grows(self):
        examples = examples_per_class()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "folds.json"
            first_folds, _ = assign_stable_folds(examples, manifest, folds=5, seed=42)
            original_assignment = {
                item[0]: index for index, fold in enumerate(first_folds) for item in fold
            }
            additions = [
                (f"new-{index}", f"novo {index}", label, f"subject:new-{index}")
                for index, label in enumerate(LABELS)
            ]
            second_folds, _ = assign_stable_folds(
                examples + additions, manifest, folds=5, seed=999
            )
            new_assignment = {
                item[0]: index for index, fold in enumerate(second_folds) for item in fold
            }

        self.assertEqual(
            original_assignment,
            {uid: new_assignment[uid] for uid in original_assignment},
        )

    def test_paired_comparison_uses_only_common_uids(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            previous_dir = results / "previous"
            previous_dir.mkdir()
            write_csv(previous_dir / "predictions.csv", [
                {"uid": "a", "label_real": LABELS[0], "label_prevista": LABELS[1]},
                {"uid": "b", "label_real": LABELS[1], "label_prevista": LABELS[1]},
            ])
            save_json(results / "latest.json", {"run_id": "previous", "run_dir": str(previous_dir)})
            paired = paired_with_previous(results, [
                {"uid": "a", "label_real": LABELS[0], "label_prevista": LABELS[0]},
                {"uid": "b", "label_real": LABELS[1], "label_prevista": LABELS[1]},
                {"uid": "new", "label_real": LABELS[2], "label_prevista": LABELS[2]},
            ])

        self.assertEqual(paired["n_common"], 2)
        self.assertEqual(paired["delta_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
