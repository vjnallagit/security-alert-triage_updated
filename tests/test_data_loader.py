import pytest
from services.data_loader import (
    DATA_DIR,
    DataFileNotFoundError,
    DataLoadError,
    MalformedDataError,
    UnexpectedStructureError,
    load_json_list,
    load_json_object,
)


def test_default_data_dir_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert len(load_json_list("scenarios.json")) == 3
    assert DATA_DIR.is_absolute()


def test_load_json_object_returns_dict():
    assert isinstance(load_json_object("experience.json"), dict)


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(DataFileNotFoundError, match="missing.json"):
        load_json_list("missing.json", tmp_path)


def test_malformed_json_raises_clear_error(tmp_path):
    (tmp_path / "bad.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(MalformedDataError, match="bad.json"):
        load_json_list("bad.json", tmp_path)


def test_object_where_list_expected_raises(tmp_path):
    (tmp_path / "obj.json").write_text("{}", encoding="utf-8")
    with pytest.raises(UnexpectedStructureError, match="array"):
        load_json_list("obj.json", tmp_path)


def test_list_where_object_expected_raises(tmp_path):
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(UnexpectedStructureError, match="object"):
        load_json_object("list.json", tmp_path)


def test_filename_cannot_escape_data_directory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "outside.json").write_text("[]", encoding="utf-8")
    with pytest.raises(DataLoadError, match="outside"):
        load_json_list("../outside.json", data_dir)
