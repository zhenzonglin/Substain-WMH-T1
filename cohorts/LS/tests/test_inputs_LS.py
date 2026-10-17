"""Synthetic LS input preparation tests; no workstation data is used."""

import copy
import csv
import importlib.util
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "vendor"))
import openpyxl

SPEC = importlib.util.spec_from_file_location("inputs_LS", PACKAGE / "inputs_LS.py")
inputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inputs)


def write_image(path, fill=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[:3, 3] = [4, -3, -2]
    image = nib.Nifti1Image(np.full((7, 8, 9), fill, dtype=np.uint8), affine)
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, str(path))


def write_clinical(path, rows=None, headers=None):
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Clinical"
    worksheet.append(headers or ["BIGDATA_CODE", "AGE", "SEX", "备注", "Original score"])
    for row in rows or [["P0015", 50.3, 1, "原始备注", 7.2], ["P0005", 21.7, 2, "保留未达常模年龄", 3], ["P0099", 50, 2, "临床额外行", 1]]:
        worksheet.append(row)
    workbook.save(str(path))


@pytest.fixture
def cohort(tmp_path):
    project = tmp_path / "GB_project"
    project.mkdir()
    raw = tmp_path / "LS_source"
    raw.mkdir()
    for participant in ("P0015", "P0005"):
        write_image(raw / participant / "scan_T1w_native.nii.gz")
        write_image(raw / participant / "scan_T2wFLAIR_native.nii.gz")
    clinical = raw / "LS_INFO.xlsx"
    write_clinical(clinical)
    template = project / "resources" / "FSL_template.nii.gz"
    write_image(template)
    reference = project / "resources" / "genmind.csv"
    reference.write_text("Age,Sex\n48,M\n50,M\n52,M\n48,F\n50,F\n52,F\n", encoding="utf-8")
    config = {
        "project_root": str(project),
        "participants": str(project / "config" / "participants_LS.tsv"),
        "ls": {"input_root": str(raw), "clinical_file": str(clinical), "preparation_dir": str(project / "inputs" / "cohort_LS"), "template": str(template)},
        "t1": {"genmind_csv": str(reference), "min_reference_n": 2, "narrow_age_half_window": 2.5, "expanded_age_half_window": 5.0},
    }
    return config


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def report_for(config):
    return json.loads((Path(config["ls"]["preparation_dir"]) / "input_report_LS.json").read_text(encoding="utf-8"))


def assert_failed(config, codes):
    with pytest.raises(inputs.InputPreparationError):
        inputs.prepare_inputs(config)
    report = report_for(config)
    assert report["status"] == "fail"
    assert set(codes).issubset({issue["code"] for issue in report["issues"]})
    assert not Path(config["participants"]).exists()
    return report


def test_complete_preparation_preserves_decimals_original_columns_and_all_people(cohort):
    raw_root = Path(cohort["ls"]["input_root"])
    before = {str(path): inputs.sha256(path) for path in raw_root.rglob("*") if path.is_file()}
    result = inputs.prepare_inputs(cohort)
    assert result["status"] == "pass"
    assert result["normative_eligible_count"] == 1
    assert result["normative_ineligible_count"] == 1
    rows = read_tsv(cohort["participants"])
    assert list(rows[0]) == list(inputs.PARTICIPANT_COLUMNS)
    assert [row["participant_id"] for row in rows] == ["P0005", "P0015"]
    assert rows[0]["sex"] == "female" and rows[1]["sex"] == "male"
    assert rows[1]["age"] == "50.3"
    assert rows[0]["age"] == "21.7"
    assert all(row["site_id"] == "LS" for row in rows)
    clinical = read_tsv(result["clinical_tsv"])
    assert len(clinical) == 3
    assert list(clinical[0]) == ["BIGDATA_CODE", "AGE", "SEX", "备注", "Original score"]
    assert clinical[0]["备注"] == "原始备注"
    assert clinical[0]["Original score"] == "7.2"
    eligibility = {row["participant_id"]: row for row in read_tsv(result["normative_eligibility_tsv"])}
    assert eligibility["P0005"]["failure_reason"] == "age_outside_22_90"
    assert eligibility["P0015"]["reference_n"] == "3"
    assert any(issue["code"] == "clinical_only" and issue["severity"] == "warning" for issue in result["issues"])
    after = {str(path): inputs.sha256(path) for path in raw_root.rglob("*") if path.is_file()}
    assert before == after
    assert result["clinical_source"]["sha256"] == before[cohort["ls"]["clinical_file"]]
    manifest = read_tsv(result["input_manifest"])
    assert len([row for row in manifest if row["source_kind"] == "original_image"]) == 4
    assert all(row["source_sha256"] == inputs.sha256(Path(row["source"])) for row in manifest)


def test_zero_mask_matches_exact_template_and_is_truthfully_marked(cohort):
    result = inputs.prepare_inputs(cohort)
    lesion = result["synthetic_lesion"]
    mask = nib.load(lesion["mask_path"])
    template = nib.load(cohort["ls"]["template"])
    assert mask.shape == template.shape
    np.testing.assert_array_equal(mask.affine, template.affine)
    assert mask.get_data_dtype() == np.dtype(np.uint8)
    assert np.count_nonzero(np.asanyarray(mask.dataobj)) == 0
    np.testing.assert_array_equal(mask.get_qform(), mask.get_sform())
    assert lesion["source_kind"] == "synthetic_zero_mask"
    assert lesion["observed_lesion_segmentation"] is False
    assert lesion["template"]["sha256"] == inputs.sha256(Path(cohort["ls"]["template"]))


def test_identical_repeat_does_not_touch_mtime(cohort):
    first = inputs.prepare_inputs(cohort)
    project = Path(cohort["project_root"])
    before = {str(path): (path.stat().st_mtime_ns, inputs.sha256(path)) for path in project.rglob("*") if path.is_file()}
    second = inputs.prepare_inputs(copy.deepcopy(cohort))
    after = {str(path): (path.stat().st_mtime_ns, inputs.sha256(path)) for path in project.rglob("*") if path.is_file()}
    assert first == second
    assert before == after


@pytest.mark.parametrize("age", ["NaN", "Inf", "-Inf", "not a number", None, 0, 121])
def test_invalid_age_blocks_without_partial_participants(cohort, age):
    write_clinical(Path(cohort["ls"]["clinical_file"]), [["P0015", age, 1], ["P0005", 50, 2]])
    assert_failed(cohort, ["invalid_age"])


@pytest.mark.parametrize("sex", [0, 3, 1.5, "male", "NaN", None, True])
def test_unknown_sex_blocks_without_truncation(cohort, sex):
    write_clinical(Path(cohort["ls"]["clinical_file"]), [["P0015", 50.3, sex], ["P0005", 50, 2]])
    assert_failed(cohort, ["invalid_sex"])


def test_reports_multiple_independent_input_errors(cohort):
    raw = Path(cohort["ls"]["input_root"])
    write_image(raw / "P0015" / "second_T1w_scan.nii.gz")
    (raw / "P0005" / "scan_T2wFLAIR_native.nii.gz").unlink()
    write_clinical(Path(cohort["ls"]["clinical_file"]), [["P0015", "Inf", 1.5], ["P0015", 51, 1]])
    report = assert_failed(cohort, ["t1w_count", "flair_count", "invalid_age", "invalid_sex", "duplicate_clinical_id", "clinical_match_missing"])
    assert report["image_subject_count"] == 2


def test_duplicate_required_headers_are_not_mangled_and_accepted(cohort):
    write_clinical(Path(cohort["ls"]["clinical_file"]), [["P0015", 50, 1, 51]], ["BIGDATA_CODE", "AGE", "SEX", " AGE "])
    assert_failed(cohort, ["duplicate_clinical_headers"])


@pytest.mark.parametrize("mode", ["multiple", "none"])
def test_requires_unique_first_row_sheet(cohort, mode):
    path = Path(cohort["ls"]["clinical_file"])
    workbook = openpyxl.load_workbook(str(path))
    if mode == "multiple":
        other = workbook.create_sheet("Second")
        other.append(["BIGDATA_CODE", "AGE", "SEX"])
        other.append(["P0015", 50, 1])
    else:
        workbook.active.insert_rows(1)
        workbook.active.cell(1, 1, "description, not the header")
    workbook.save(str(path))
    assert_failed(cohort, ["clinical_sheet_count"])


def test_formula_without_cache_is_blocking_in_any_column(cohort):
    path = Path(cohort["ls"]["clinical_file"])
    workbook = openpyxl.load_workbook(str(path))
    workbook.active["E2"] = "=1+2"
    workbook.save(str(path))
    assert_failed(cohort, ["formula_cache_missing"])
    formulas = read_tsv(Path(cohort["ls"]["preparation_dir"]) / "clinical_formulas_LS.tsv")
    assert formulas[0]["formula"] == "=1+2"


def test_cached_formula_uses_decimal_cache_and_records_formula(cohort):
    path = Path(cohort["ls"]["clinical_file"])
    workbook = openpyxl.load_workbook(str(path))
    workbook.active["B2"] = "=50+0.3"
    workbook.save(str(path))
    with zipfile.ZipFile(str(path)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    xml = ET.fromstring(members["xl/worksheets/sheet1.xml"])
    cell = xml.find(".//m:c[@r='B2']", namespace)
    assert cell.find("m:f", namespace).text == "50+0.3"
    cell.find("m:v", namespace).text = "50.3"
    members["xl/worksheets/sheet1.xml"] = ET.tostring(xml, encoding="utf-8")
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    result = inputs.prepare_inputs(cohort)
    assert read_tsv(result["clinical_tsv"])[0]["AGE"] == "50.3"
    assert next(row for row in read_tsv(cohort["participants"]) if row["participant_id"] == "P0015")["age"] == "50.3"


def test_reference_missing_is_blocking(cohort):
    Path(cohort["t1"]["genmind_csv"]).unlink()
    assert_failed(cohort, ["normative_reference_error"])


def test_reference_insufficiency_keeps_all_participants(cohort):
    cohort["t1"]["min_reference_n"] = 300
    result = inputs.prepare_inputs(cohort)
    assert result["status"] == "pass"
    assert result["normative_eligible_count"] == 0
    assert len(read_tsv(cohort["participants"])) == 2
    eligibility = read_tsv(result["normative_eligibility_tsv"])
    assert any(row["failure_reason"] == "reference_n_below_300" for row in eligibility)


def test_no_id_zero_padding_or_automatic_case_conversion(cohort):
    write_clinical(Path(cohort["ls"]["clinical_file"]), [["P15", 50, 1], ["p0005", 50, 2]])
    report = assert_failed(cohort, ["invalid_clinical_id", "clinical_match_missing"])
    assert any(issue["code"] == "clinical_only" and issue["participant_id"] == "P15" for issue in report["issues"])


def test_failed_reprepare_preserves_previously_published_participants(cohort):
    inputs.prepare_inputs(cohort)
    path = Path(cohort["participants"])
    original = path.read_bytes(), path.stat().st_mtime_ns
    write_clinical(Path(cohort["ls"]["clinical_file"]), [["P0015", 50, 3], ["P0005", 50, 2]])
    with pytest.raises(inputs.InputPreparationError):
        inputs.prepare_inputs(cohort)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert report_for(cohort)["status"] == "fail"


def test_refuses_source_destination_overlap(cohort):
    cohort["ls"]["preparation_dir"] = cohort["ls"]["input_root"]
    with pytest.raises(inputs.InputPreparationError, match="must not overlap"):
        inputs.prepare_inputs(cohort)


def test_invalid_existing_zero_mask_is_reported_not_replaced(cohort):
    inputs.prepare_inputs(cohort)
    path = Path(cohort["ls"]["preparation_dir"]) / inputs.MASK_NAME
    write_image(path, fill=1)
    before = inputs.sha256(path)
    with pytest.raises(inputs.InputPreparationError):
        inputs.prepare_inputs(cohort)
    assert inputs.sha256(path) == before
    assert any(issue["code"] == "existing_zero_mask_error" for issue in report_for(cohort)["issues"])


def test_corrupt_xlsx_produces_report(cohort):
    Path(cohort["ls"]["clinical_file"]).write_bytes(b"not an XLSX archive")
    assert_failed(cohort, ["clinical_read_error"])


def test_output_symlink_does_not_modify_target(cohort, tmp_path):
    target = tmp_path / "unrelated.tsv"
    target.write_text("do not modify", encoding="utf-8")
    participants = Path(cohort["participants"])
    participants.parent.mkdir(parents=True)
    participants.symlink_to(target)
    with pytest.raises(inputs.InputPreparationError, match="Unsafe LS output"):
        inputs.prepare_inputs(cohort)
    assert target.read_text(encoding="utf-8") == "do not modify"


def test_changing_workbook_during_read_is_not_published(cohort, monkeypatch):
    original = inputs._read_clinical

    def changing_reader(path, issues):
        result = original(path, issues)
        workbook = openpyxl.load_workbook(str(path))
        workbook.active["B2"] = 55.7
        workbook.save(str(path))
        return result

    monkeypatch.setattr(inputs, "_read_clinical", changing_reader)
    assert_failed(cohort, ["clinical_changed_during_read"])
