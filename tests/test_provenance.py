import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gravitas.registry.provenance import get_registry_provenance


def test_registry_is_non_empty_and_every_entry_has_a_source():
    prov = get_registry_provenance()
    assert len(prov) > 10
    assert all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in prov.items())
