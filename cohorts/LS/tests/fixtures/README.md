# Source fixtures for LS regression tests

These files reproduce the original, GB, and GB concurrency-18 source shapes used
by the adapter's regression suite. They are test inputs only. The installer never
copies them into the working analysis project.

The fixture contains 37 source/configuration files:

- `Substain-v1.0.9-remote-sim`: 19 Python modules, 7 stage shell scripts, and
  7 configuration/workflow/launcher files required by `fixture_project`.
- `Substain-ws1-rolling-patch-release`: the GB patch planner and mocked GPU harness.
- `deliverables`: the concurrency-18 patch planner and its audit helper.

Source bytes are preserved except for CRLF-to-LF normalization. No participant
table, clinical workbook, image, model, log, or runtime environment is included.
Tests generate synthetic participant records, tiny images, and mocked GPU output
inside pytest temporary directories. `install_offline.sh` is an archived source
fixture required to build the historical overlay; it is never executed or offered
as the LS Git installation route. Pytest does not separately collect tests inside
this fixture directory.
