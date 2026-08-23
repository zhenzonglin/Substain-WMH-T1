import numpy as np

from substain_features.symmetry import (
    compose_native_wmh,
    contralateral_rule_in_common_space,
    label_lesion_components,
    mirror_world_x_zero,
)


def _centered_affine() -> np.ndarray:
    affine = np.eye(4)
    affine[0, 3] = -4.0
    return affine


def test_world_reflection_uses_physical_x_zero() -> None:
    data = np.zeros((9, 5, 5), dtype=np.uint8)
    data[2, 2, 2] = 1  # world x=-2 mm
    mirrored, valid = mirror_world_x_zero(data, _centered_affine())
    assert mirrored[6, 2, 2]  # world x=+2 mm
    assert valid[4, 2, 2]


def test_lesion_components_use_26_neighbourhood() -> None:
    lesion = np.zeros((5, 5, 5), dtype=np.uint8)
    lesion[1, 1, 1] = 1
    lesion[2, 2, 2] = 1
    labels, count = label_lesion_components(lesion)
    assert count == 1
    assert labels[1, 1, 1] == labels[2, 2, 2]


def test_unilateral_lesion_receives_contralateral_wmh() -> None:
    shape = (9, 7, 7)
    wmh = np.zeros(shape, dtype=np.uint8)
    lesion = np.zeros(shape, dtype=np.uint8)
    components = np.zeros(shape, dtype=np.int16)
    brain = np.ones(shape, dtype=np.uint8)
    wmh[2, 3, 3] = 1
    wmh[6, 3, 3] = 1
    lesion[6, 3, 3] = 1
    components[6, 3, 3] = 1
    result = contralateral_rule_in_common_space(
        wmh, lesion, components, brain, _centered_affine(), [1]
    )
    assert result["donor"][6, 3, 3]
    assert result["replacement"][6, 3, 3]


def test_bilateral_symmetric_lesions_remove_conflicting_donor() -> None:
    shape = (9, 7, 7)
    wmh = np.zeros(shape, dtype=np.uint8)
    lesion = np.zeros(shape, dtype=np.uint8)
    components = np.zeros(shape, dtype=np.int16)
    brain = np.ones(shape, dtype=np.uint8)
    for index, component in ((2, 1), (6, 2)):
        wmh[index, 3, 3] = 1
        lesion[index, 3, 3] = 1
        components[index, 3, 3] = component
    result = contralateral_rule_in_common_space(
        wmh, lesion, components, brain, _centered_affine(), [1, 2]
    )
    assert result["conflict"][2, 3, 3]
    assert result["conflict"][6, 3, 3]
    assert not result["replacement"].any()


def test_partial_bilateral_conflict_removes_only_conflicting_voxel() -> None:
    shape = (9, 7, 7)
    wmh = np.zeros(shape, dtype=np.uint8)
    lesion = np.zeros(shape, dtype=np.uint8)
    components = np.zeros(shape, dtype=np.int16)
    brain = np.ones(shape, dtype=np.uint8)
    wmh[2, 3:5, 3] = 1
    wmh[6, 3:5, 3] = 1
    lesion[6, 3:5, 3] = 1
    lesion[2, 3, 3] = 1
    components[6, 3:5, 3] = 1
    components[2, 3, 3] = 2
    result = contralateral_rule_in_common_space(
        wmh, lesion, components, brain, _centered_affine(), [1]
    )
    assert result["conflict"][6, 3, 3]
    assert not result["replacement"][6, 3, 3]
    assert result["replacement"][6, 4, 3]


def test_native_composition_never_changes_wmh_outside_triggered_lesion() -> None:
    original = np.zeros((7, 7, 7), dtype=np.uint8)
    original[1, 1, 1] = 1
    original[5, 5, 5] = 1
    triggered = np.zeros_like(original)
    triggered[5, 5, 5] = 1
    replacement = np.zeros_like(original)
    result = compose_native_wmh(original, triggered, replacement)
    np.testing.assert_array_equal(result[~triggered.astype(bool)], original[~triggered.astype(bool)])
    assert not result[5, 5, 5]


def test_lesion_component_without_wmh_overlap_does_not_trigger() -> None:
    shape = (9, 7, 7)
    wmh = np.zeros(shape, dtype=np.uint8)
    lesion = np.zeros(shape, dtype=np.uint8)
    components = np.zeros(shape, dtype=np.int16)
    brain = np.ones(shape, dtype=np.uint8)
    wmh[2, 3, 3] = 1
    lesion[6, 3, 3] = 1
    components[6, 3, 3] = 1
    result = contralateral_rule_in_common_space(
        wmh, lesion, components, brain, _centered_affine(), []
    )
    assert not result["triggered"].any()
    assert not result["replacement"].any()
