import json
from pathlib import Path

import pandas as pd

from substain_features.pipeline import export_reviewed_outputs
from substain_features.qc_review import (
    _connect,
    _html,
    initialise_reviews,
    load_review_table,
    qc_figure_paths,
    review_database,
)
from substain_features.schema import Participant


def _participant(tmp_path: Path, participant_id: str = "A01") -> Participant:
    return Participant(
        participant_id,
        60.0,
        "female",
        "SITE",
        tmp_path / "t1.nii.gz",
        tmp_path / "flair.nii.gz",
        tmp_path / "lesion.nii.gz",
    )


def _four_images(qc_dir: Path, participant_id: str) -> None:
    for index, path in enumerate(qc_figure_paths(qc_dir, participant_id).values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png" + bytes([index]))


def test_qc_review_resumes_and_invalidates_changed_images(tmp_path: Path) -> None:
    participant = _participant(tmp_path)
    derivatives = tmp_path / "derivatives"
    qc_dir = derivatives / "qc"
    _four_images(qc_dir, participant.participant_id)
    initialise_reviews([participant], qc_dir, derivatives)
    with _connect(review_database(qc_dir)) as connection:
        connection.execute(
            "UPDATE reviews SET review_state='fail',reasons_json=?,note='检查记录' WHERE participant_id=?",
            (json.dumps(["registration_invalid"]), participant.participant_id),
        )
        connection.commit()
    unchanged = load_review_table([participant], qc_dir, derivatives)
    assert unchanged.loc[0, "review_state"] == "fail"
    qc_figure_paths(qc_dir, participant.participant_id)["lesion_on_T1"].write_bytes(b"updated")
    changed = load_review_table([participant], qc_dir, derivatives)
    assert changed.loc[0, "review_state"] == "stale"


def test_qc_gui_contains_exactly_four_panels_and_multiple_failure_reasons() -> None:
    html = _html().decode("utf-8")
    for kind in ("lesion_on_T1", "lesion_on_FLAIR", "WMH_lesion_overlap", "T1_macro20"):
        assert kind in html
    for reason in ("t1_invalid", "flair_invalid", "registration_invalid", "wmh_failed", "macro_failed"):
        assert 'value="{}"'.format(reason) in html
    assert "qc_pass" in html


def test_primary40_export_only_keeps_manual_qc_pass(tmp_path: Path) -> None:
    participants = [_participant(tmp_path, "A01"), _participant(tmp_path, "A02")]
    derivatives = tmp_path / "derivatives"
    tables = derivatives / "tables"
    qc_dir = derivatives / "qc"
    tables.mkdir(parents=True)
    columns = ["participant_id"] + ["feature_{:02d}".format(index) for index in range(40)]
    pd.DataFrame([["A01"] + list(range(40)), ["A02"] + list(range(40))], columns=columns).to_csv(
        tables / "features_computed40.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            {"participant_id": "A01", "multimodal_ineligible": False},
            {"participant_id": "A02", "multimodal_ineligible": False},
        ]
    ).to_csv(tables / "subject_qc.tsv", sep="\t", index=False)
    for participant in participants:
        _four_images(qc_dir, participant.participant_id)
    initialise_reviews(participants, qc_dir, derivatives)
    with _connect(review_database(qc_dir)) as connection:
        connection.execute(
            "UPDATE reviews SET review_state='pass',qc_pass=1,reasons_json='[]' WHERE participant_id='A01'"
        )
        connection.execute(
            "UPDATE reviews SET review_state='fail',qc_pass=0,reasons_json=? WHERE participant_id='A02'",
            (json.dumps(["macro_failed"]),),
        )
        connection.commit()
    config = {
        "project_root": str(tmp_path),
        "derivatives": "derivatives",
        "qc_dir": "derivatives/qc",
    }
    result = export_reviewed_outputs(config, participants)
    primary = pd.read_csv(tables / "features_primary40.tsv", sep="\t")
    assert result["qc_pass_subjects"] == 1
    assert primary["participant_id"].tolist() == ["A01"]
    subject_qc = pd.read_csv(tables / "subject_qc.tsv", sep="\t")
    assert subject_qc["manual_qc_pass"].tolist() == [True, False]
