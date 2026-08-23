from pathlib import Path

import pandas as pd
import pytest

from substain_features.input_prep import prepare_inputs


def _config(root: Path) -> dict:
    return {
        "project_root": str(root),
        "participants": "config/participants.tsv",
        "input": {
            "mode": "folders",
            "t1_root": "incoming/t1",
            "flair_root": "incoming/flair",
            "lesion_root": "incoming/lesion",
            "t1_suffix": "_T1w.nii.gz",
            "flair_suffix": "_FLAIR.nii.gz",
            "lesion_suffix": "_lesion.nii.gz",
            "metadata_tsv": "config/metadata.tsv",
            "bids_links": "inputs/bids_links",
        },
    }


def test_recursive_folder_matching_builds_sessionless_bids_links(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "metadata.tsv").write_text(
        "participant_id\tage\tsex\tsite_id\nA01\t60\tfemale\tSITE\n", encoding="utf-8"
    )
    paths = {
        "t1": tmp_path / "incoming" / "t1" / "level1" / "A01_T1w.nii.gz",
        "flair": tmp_path / "incoming" / "flair" / "level1" / "level2" / "A01_FLAIR.nii.gz",
        "lesion": tmp_path / "incoming" / "lesion" / "A01_lesion.nii.gz",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
    result = prepare_inputs(config)
    assert result["participant_count"] == 1
    t1_link = tmp_path / "inputs" / "bids_links" / "sub-A01" / "anat" / "sub-A01_T1w.nii.gz"
    flair_link = tmp_path / "inputs" / "bids_links" / "sub-A01" / "anat" / "sub-A01_FLAIR.nii.gz"
    assert t1_link.is_symlink() and t1_link.resolve() == paths["t1"].resolve()
    assert flair_link.is_symlink() and flair_link.resolve() == paths["flair"].resolve()
    table = pd.read_csv(tmp_path / "config" / "participants.tsv", sep="\t")
    assert table["participant_id"].tolist() == ["A01"]
    assert "ses-" not in table.loc[0, "t1w"]


def test_prepare_inputs_does_not_touch_unchanged_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "metadata.tsv").write_text(
        "participant_id\tage\tsex\tsite_id\nA01\t60\tfemale\tSITE\n", encoding="utf-8"
    )
    for relative in (
        "incoming/t1/A01_T1w.nii.gz",
        "incoming/flair/A01_FLAIR.nii.gz",
        "incoming/lesion/A01_lesion.nii.gz",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    first = prepare_inputs(config)
    participant_mtime = (tmp_path / "config" / "participants.tsv").stat().st_mtime_ns
    second = prepare_inputs(config)
    assert first["participants_tsv_changed"] is True
    assert second["participants_tsv_changed"] is False
    assert second["input_manifest_changed"] is False
    assert (tmp_path / "config" / "participants.tsv").stat().st_mtime_ns == participant_mtime


def test_recursive_folder_matching_rejects_duplicate_id(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "metadata.tsv").write_text(
        "participant_id\tage\tsex\tsite_id\nA01\t60\tmale\tSITE\n", encoding="utf-8"
    )
    for relative in (
        "incoming/t1/a/A01_T1w.nii.gz",
        "incoming/t1/b/A01_T1w.nii.gz",
        "incoming/flair/A01_FLAIR.nii.gz",
        "incoming/lesion/A01_lesion.nii.gz",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    with pytest.raises(ValueError, match="重复ID"):
        prepare_inputs(config)


def test_bids_multiple_sessions_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["input"].update({"mode": "bids", "bids_root": "BIDS"})
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "metadata.tsv").write_text(
        "participant_id\tage\tsex\tsite_id\nA01\t60\tfemale\tSITE\n", encoding="utf-8"
    )
    lesion = tmp_path / "incoming" / "lesion" / "A01_lesion.nii.gz"
    lesion.parent.mkdir(parents=True)
    lesion.write_bytes(b"x")
    for session in ("01", "02"):
        anat = tmp_path / "BIDS" / "sub-A01" / "ses-{}".format(session) / "anat"
        anat.mkdir(parents=True)
        (anat / "sub-A01_ses-{}_T1w.nii.gz".format(session)).write_bytes(b"t1")
        (anat / "sub-A01_ses-{}_FLAIR.nii.gz".format(session)).write_bytes(b"flair")
    with pytest.raises(ValueError, match="多个BIDS session"):
        prepare_inputs(config)
