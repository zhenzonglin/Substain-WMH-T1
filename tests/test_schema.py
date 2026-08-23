from pathlib import Path

import pytest

from substain_features.schema import load_participants


def test_participants_reject_ambiguous_sex(tmp_path: Path) -> None:
    manifest = tmp_path / "participants.tsv"
    manifest.write_text(
        "participant_id\tage\tsex\tsite_id\tt1w\tflair\tlesion_mask\n"
        "A\t60\t1\tX\ta.nii.gz\tb.nii.gz\tc.nii.gz\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="female/male"):
        load_participants(manifest, tmp_path)


def test_participants_reject_legacy_session_column(tmp_path: Path) -> None:
    manifest = tmp_path / "participants.tsv"
    manifest.write_text(
        "participant_id\tsession_id\tage\tsex\tsite_id\tt1w\tflair\tlesion_mask\n"
        "A\tses01\t60\tfemale\tX\ta.nii.gz\tb.nii.gz\tc.nii.gz\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extra=.*session_id"):
        load_participants(manifest, tmp_path)
