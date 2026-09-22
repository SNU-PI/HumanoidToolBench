"""Release snapshots contain only approved sources and never replace outputs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_public_evaluation.py"
SPEC = importlib.util.spec_from_file_location("public_export", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_release_source_contains_only_exported_python_modules():
    root = SCRIPT.parents[1]
    exported = {
        root / path
        for path in exporter.selected_files(root)
        if path.parts[0] == "src" and path.suffix == ".py"
    }
    actual = set((root / "src").rglob("*.py"))
    assert actual == exported


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for name in exporter.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"public {name}\n")
    for name in exporter.SOURCE_DIRS:
        path = root / name / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# public runtime\n")
    path = root / exporter.ASSET_DIR / "catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]\n")
    return root


def test_export_excludes_private_data_training_variants_and_git(source, tmp_path):
    private = (
        ".env",
        ".git/config",
        "artifacts/result.json",
        "paper/main.tex",
        "data/private_recording.parquet",
        "third_party/gear_sonic/.env",
        "src/theta_bench/cli/train_private.py",
        "src/theta_bench/teleop/private.py",
        "src/theta_bench/evals/easy.py",
        "src/theta_bench/assets/.env",
        "src/theta_bench/assets/__pycache__/private.pyc",
    )
    for name in private:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private secret marker")
    output = tmp_path / "release"
    manifest = exporter.export_evaluation(output, source=source)

    for name in private:
        assert not (output / name).exists()
    assert (output / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    assert not (output / "third_party").exists()
    for name, receipt in manifest["files"].items():
        content = (output / name).read_bytes()
        assert b"private secret marker" not in content
        assert receipt == {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    assert json.loads((output / "export_manifest.json").read_text()) == manifest
    assert str(source) not in (output / "export_manifest.json").read_text()


def test_export_refuses_nonempty_output_without_changing_it(source, tmp_path):
    output = tmp_path / "release"
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("keep")
    with pytest.raises(ValueError, match="must be empty"):
        exporter.export_evaluation(output, source=source)
    assert list(output.iterdir()) == [existing]
    assert existing.read_text() == "keep"


def test_export_accepts_an_existing_empty_directory(source, tmp_path):
    output = tmp_path / "release"
    output.mkdir()
    exporter.export_evaluation(output, source=source)
    assert (output / "README.md").is_file()


def test_missing_required_source_fails_before_creating_output(source, tmp_path):
    (source / "LICENSE").unlink()
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Release file is missing"):
        exporter.export_evaluation(output, source=source)
    assert not output.exists()


@pytest.mark.parametrize("link_source", [True, False])
def test_export_rejects_symlinks_without_copying_private_content(
    source, tmp_path, link_source
):
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret").write_text("private")
    output = tmp_path / "release"
    if link_source:
        link = source / "src/theta_bench/assets/linked"
        link.symlink_to(private, target_is_directory=True)
    else:
        output.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        exporter.export_evaluation(output, source=source)
    assert list(private.iterdir()) == [private / "secret"]
