"""Reference source provenance is part of a reproducible public checkout."""
import hashlib
import json
from flash_attention_lab.registry import ROOT


def test_vendored_reference_manifest_and_licenses():
    groups = json.loads((ROOT / 'references/manifest.json').read_text())
    assert len(groups) == 2
    for group in groups:
        assert len(group['commit']) == 40
        assert group['sha256']
        assert any('LICENSE' in name for name in group['sha256'])
        for name, digest in group['sha256'].items():
            source = ROOT / 'references' / name
            assert source.is_file(), name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name
