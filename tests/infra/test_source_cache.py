"""FA4 patches produce isolated, reproducible compiler inputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import pytest
from flash_attention_lab import source_cache
from flash_attention_lab.errors import EnvironmentUnsupported


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    upstream = tmp_path / 'third_party/flash-attention'
    package = upstream / 'flash_attn/cute'
    package.mkdir(parents=True)
    source = package / 'model.py'
    source.write_text('threshold = 8\n')
    patch = tmp_path / 'patches/flash-attention/rescale-threshold.patch'
    patch.parent.mkdir(parents=True)
    patch.write_text('--- a/flash_attn/cute/model.py\n+++ b/flash_attn/cute/model.py\n@@ -1 +1 @@\n-threshold = 8\n+threshold = 0\n')
    monkeypatch.setattr(source_cache, 'ROOT', tmp_path)
    monkeypatch.setattr(source_cache, 'require_source', lambda name: upstream)
    # Exercise git apply inside an enclosing repo, not just a standalone temp dir.
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    return source, patch


def test_patch_only_changes_cached_copy_and_tracks_all_inputs(source_tree):
    source, patch = source_tree
    first = source_cache.prepare_flash_attention_source()
    assert (first / 'flash_attn/cute/model.py').read_text() == 'threshold = 0\n'
    assert source.read_text() == 'threshold = 8\n'
    assert source_cache.prepare_flash_attention_source() == first
    patch.write_text(patch.read_text().replace('+threshold = 0', '+threshold = 4'))
    second = source_cache.prepare_flash_attention_source()
    assert second != first
    assert (second / 'flash_attn/cute/model.py').read_text() == 'threshold = 4\n'
    (source.parent / 'helper.py').write_text('value = 1\n')
    assert source_cache.prepare_flash_attention_source() != second


def test_incompatible_patch_leaves_no_published_cache(source_tree):
    source, _ = source_tree
    source.write_text('changed_upstream = True\n')
    with pytest.raises(EnvironmentUnsupported, match='patch cannot be applied'):
        source_cache.prepare_flash_attention_source()
    assert not list((source_cache.ROOT / '.cache/sources/flash-attention').iterdir())
    assert source.read_text() == 'changed_upstream = True\n'


def test_concurrent_preparation_publishes_complete_identical_source(source_tree):
    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(lambda _: source_cache.prepare_flash_attention_source(), range(3)))
    assert len(set(results)) == 1
    assert (results[0] / 'flash_attn/cute/model.py').read_text() == 'threshold = 0\n'


def test_modified_compiler_input_is_rejected(source_tree):
    target = source_cache.prepare_flash_attention_source()
    (target / 'flash_attn/cute/model.py').write_text('unexpected = True\n')
    with pytest.raises(EnvironmentUnsupported, match='cache is incomplete or modified'):
        source_cache.prepare_flash_attention_source()


def test_edit_during_preparation_cannot_change_cached_snapshot(source_tree, monkeypatch):
    from contextlib import contextmanager
    source, patch = source_tree
    initial_patch = patch.read_text()
    temporary_directory = source_cache.tempfile.TemporaryDirectory

    @contextmanager
    def edit_after_snapshot(*args, **kwargs):
        with temporary_directory(*args, **kwargs) as directory:
            source.write_text('threshold = 12\n')
            patch.write_text(initial_patch.replace('-threshold = 8', '-threshold = 12')
                             .replace('+threshold = 0', '+threshold = 6'))
            yield directory

    monkeypatch.setattr(source_cache.tempfile, 'TemporaryDirectory', edit_after_snapshot)
    result = source_cache.prepare_flash_attention_source()
    assert (result / 'flash_attn/cute/model.py').read_text() == 'threshold = 0\n'
    source.write_text('threshold = 8\n')
    patch.write_text(initial_patch)
    assert source_cache.prepare_flash_attention_source() == result
