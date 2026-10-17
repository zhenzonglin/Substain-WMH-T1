"""Two real pool implementations contend for the same mocked physical GPUs."""
import sys

from test_overlay_LS import WORKSPACE, fixture_project, load_file, build_overlay


def test_gb_and_ls_share_the_same_two_card_admission_locks(tmp_path):
    root, config, _ = fixture_project(tmp_path / 'project', 'gb18')
    build_overlay(root, config, root / 'runtime_LS')
    sys.path.insert(0, str(WORKSPACE / 'Substain-ws1-rolling-patch-release'))
    module = load_file('ls_cross_cohort_gpu_harness', WORKSPACE / 'Substain-ws1-rolling-patch-release/test_gb_minimal_gpu.py')
    harness_root = tmp_path / 'harness'
    harness_root.mkdir()
    harness = module.Harness(harness_root)
    gb_pool = root / 'src/substain_features/gpu_pool.py'
    ls_pool = root / 'runtime_LS/src/substain_features/gpu_pool.py'
    assert gb_pool.read_bytes() == ls_pool.read_bytes()
    try:
        harness.pool_file = gb_pool
        gb = harness.launch('gb_wmh', wmh=True)
        assert harness.started('gb_wmh')['device'] == '0'
        harness.pool_file = ls_pool
        ls = harness.launch('ls_wmh', wmh=True)
        assert harness.started('ls_wmh')['device'] == '1'
        fallback = harness.launch('ls_t1')
        assert harness.started('ls_t1')['profile'] == 'cpu'
        harness.release('ls_t1', fallback)
        harness.release('ls_wmh', ls)
        harness.release('gb_wmh', gb)
    finally:
        harness.close()
