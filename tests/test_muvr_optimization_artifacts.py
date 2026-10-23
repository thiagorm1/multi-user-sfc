import pytest

from scripts.report_muvr_optimization import read_log, save_new
from scripts.run_muvr_final_evaluations import write_new


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_windows_validation_logs_decode_without_losing_evidence(tmp_path, encoding):
    path = tmp_path / "pytest.log"
    expected = "128 passed in 12.44s\n"
    path.write_bytes(expected.encode(encoding))
    assert read_log(path) == expected


def test_audit_and_runner_never_overwrite_existing_files(tmp_path):
    path = tmp_path / "evidence.json"
    write_new(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"original": False})
    with pytest.raises(FileExistsError):
        save_new(path, "replacement")
    assert path.read_bytes() == before
