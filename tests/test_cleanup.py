from pathlib import Path

from substain_features.pipeline import stage_cleanup, status_path
from substain_features.schema import Participant
from substain_features.status import write_status


def test_success_cleanup_keeps_final_outputs_and_removes_rebuildable_files(tmp_path: Path) -> None:
    config = {"project_root": str(tmp_path), "derivatives": "derivatives"}
    participant = Participant("A01", 60.0, "female", "SITE", tmp_path / "t1", tmp_path / "flair", tmp_path / "lesion")
    subject = tmp_path / "derivatives" / "sub-A01"

    def make(relative: str, content: bytes = b"x") -> str:
        path = subject / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    kept = {
        "lesion_t1": make("lesion/lesion_t1.nii.gz"),
        "lesion_flair": make("lesion/lesion_flair.nii.gz"),
        "seg": make("wmh/seg.nii.gz"),
        "original": make("wmh/contralateral/original.nii.gz"),
        "corrected": make("wmh/contralateral/corrected.nii.gz"),
        "wmh_features": make("wmh/features.json"),
        "t1_seg": make("t1/tool/final.nii.gz"),
        "macro": make("t1/macro.nii.gz"),
        "t1_features": make("t1/features.json"),
    }
    removable = {
        "probability": make("wmh/probability.nii.gz"),
        "warp": make("registration/t1_to_ch2better_1Warp.nii.gz"),
        "lesion_ch2better": make("lesion/lesion_ch2better.nii.gz"),
        "donor": make("wmh/contralateral/donor.nii.gz"),
    }
    make("t1/nichart_tool_output/temp_working_dir/scratch.nii.gz")
    write_status(status_path(config, participant, "registration"), "registration", "pass", "A01", {})
    write_status(
        status_path(config, participant, "lesion"), "lesion", "pass", "A01",
        {"lesion_t1": kept["lesion_t1"], "lesion_flair": kept["lesion_flair"], "lesion_ch2better": removable["lesion_ch2better"]},
    )
    write_status(
        status_path(config, participant, "wmh_seg"), "wmh_seg", "pass", "A01",
        {"segmentation": kept["seg"], "probability_map": removable["probability"]},
    )
    write_status(
        status_path(config, participant, "wmh"), "wmh", "pass", "A01",
        {"original_wmh": kept["original"], "corrected_wmh": kept["corrected"], "feature_json": kept["wmh_features"], "donor_native_flair": removable["donor"]},
    )
    write_status(
        status_path(config, participant, "t1"), "t1", "pass", "A01",
        {"segmentation": kept["t1_seg"], "macro20_segmentation": kept["macro"], "feature_json": kept["t1_features"]},
    )
    write_status(status_path(config, participant, "qc"), "qc", "pass", "A01", {"figure_count": 4})
    details = stage_cleanup(config, participant)
    assert all(Path(path).is_file() for path in kept.values())
    assert all(not Path(path).exists() for path in removable.values())
    assert not (subject / "t1/nichart_tool_output/temp_working_dir").exists()
    assert details["bytes_removed"] > 0
