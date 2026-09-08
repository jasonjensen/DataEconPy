"""File handling, the mapping protocol, attributes, searching and error behaviour."""

from __future__ import annotations

import numpy as np
import pytest

import dataecon as de
from dataecon._consts import Class, Type


class TestOpening:
    def test_opening_for_write_creates_the_file(self, daec_path) -> None:
        with de.opendaec(daec_path, write=True) as f:
            f.write("x", 1.0)
        assert daec_path.exists()

    def test_default_is_read_only(self, daec_path) -> None:
        de.writedb(daec_path, {"x": 1.0})
        with de.opendaec(daec_path) as f:
            assert f.readonly
            with pytest.raises(de.DEReadOnlyError, match="read-only"):
                f.write("y", 2.0)

    def test_a_refused_write_leaves_the_file_usable(self, daec_path) -> None:
        """A read-only write must not wedge the handle.

        Reaching the C library with a doomed write opens a transaction that can
        never commit, after which the file cannot even be closed. The read-only
        check happens before that, so the handle stays usable.
        """
        de.writedb(daec_path, {"x": 1.0})
        with de.opendaec(daec_path) as f:
            with pytest.raises(de.DEReadOnlyError):
                f.write("y", 2.0)
            assert f.read("x") == 1.0
            with pytest.raises(de.DEReadOnlyError):
                f.set_attribute("x", "k", "v")
            with pytest.raises(de.DEReadOnlyError):
                f.delete_object("x")
        assert f.closed

    def test_truncate_needs_write(self, daec_path) -> None:
        de.writedb(daec_path, {"x": 1.0})
        with pytest.raises(de.DEArgumentError, match="read-only"):
            de.opendaec(daec_path, truncate=True)

    def test_truncate_empties_the_file(self, daec_path) -> None:
        de.writedb(daec_path, {"x": 1.0, "y": 2.0})
        with de.opendaec(daec_path, write=True, truncate=True) as f:
            assert f.keys() == []

    def test_close_is_idempotent(self, daec_path) -> None:
        f = de.opendaec(daec_path, write=True)
        f.close()
        f.close()
        assert f.closed

    def test_using_a_closed_file_raises(self, daec_path) -> None:
        f = de.opendaec(daec_path, write=True)
        f.close()
        with pytest.raises(de.DEError, match="closed"):
            f.keys()

    def test_context_manager_closes(self, daec_path) -> None:
        with de.opendaec(daec_path, write=True) as f:
            pass
        assert f.closed

    def test_repr_says_the_state(self, daec_path) -> None:
        with de.opendaec(daec_path, write=True) as f:
            assert "read-write" in repr(f)
        assert "closed" in repr(f)

    def test_memory_file_is_writable_and_empty(self) -> None:
        with de.opendaecmem() as f:
            assert not f.readonly
            assert f.keys() == []


class TestMappingProtocol:
    def test_setitem_getitem(self, memfile: de.DEFile) -> None:
        memfile["x"] = 1.5
        assert memfile["x"] == 1.5

    def test_setitem_overwrites(self, memfile: de.DEFile) -> None:
        memfile["x"] = 1.5
        memfile["x"] = 2.5
        assert memfile["x"] == 2.5

    def test_contains(self, memfile: de.DEFile) -> None:
        memfile["x"] = 1.0
        assert "x" in memfile
        assert "y" not in memfile

    def test_delitem(self, memfile: de.DEFile) -> None:
        memfile["x"] = 1.0
        del memfile["x"]
        assert "x" not in memfile

    def test_len_and_keys_and_iter(self, memfile: de.DEFile) -> None:
        memfile["a"] = 1.0
        memfile["b"] = 2.0
        assert len(memfile) == 2
        assert sorted(memfile.keys()) == ["a", "b"]
        assert sorted(memfile) == ["a", "b"]

    def test_nested_path_access(self, memfile: de.DEFile) -> None:
        memfile["a/b/c"] = 3.0
        assert memfile["a/b/c"] == 3.0
        assert memfile["/a/b/c"] == 3.0


class TestOverwrite:
    def test_storing_over_an_existing_name_raises_by_default(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        with pytest.raises(de.DEExistsError):
            memfile.write("x", 2.0)

    def test_overwrite_flag_on_the_file(self) -> None:
        with de.opendaecmem(overwrite=True) as f:
            f.write("x", 1.0)
            f.write("x", 2.0)
            assert f.read("x") == 2.0

    def test_overwrite_argument_on_the_call(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.write("x", 2.0, overwrite=True)
        assert memfile.read("x") == 2.0

    def test_overwrite_changes_the_storage_class(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.write("x", np.arange(3.0), overwrite=True)
        assert np.array_equal(memfile.read("x"), np.arange(3.0))


class TestAttributes:
    def test_set_and_get(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.set_attribute("x", "units", "dollars")
        assert memfile.get_attribute("x", "units") == "dollars"

    def test_missing_attribute_returns_the_default(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        assert memfile.get_attribute("x", "nope") is None
        assert memfile.get_attribute("x", "nope", "fallback") == "fallback"

    def test_missing_attribute_can_raise(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        with pytest.raises(de.DEMissingAttributeError):
            memfile.get_attribute("x", "nope", required=True)

    def test_setting_twice_replaces(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.set_attribute("x", "k", "one")
        memfile.set_attribute("x", "k", "two")
        assert memfile.get_attribute("x", "k") == "two"

    def test_get_all_attributes(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.set_attribute("x", "a", "1")
        memfile.set_attribute("x", "b", "2")
        assert memfile.get_all_attributes("x") == {"a": "1", "b": "2"}

    def test_get_all_attributes_of_an_object_with_none(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        assert memfile.get_all_attributes("x") == {}

    def test_a_value_containing_the_delimiter_is_reported_clearly(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.set_attribute("x", "a", "one|two")
        memfile.set_attribute("x", "b", "three")
        with pytest.raises(de.DEError, match="delimiter"):
            memfile.get_all_attributes("x", delimiter="|")

    def test_attributes_survive_a_file_round_trip(self, daec_path) -> None:
        with de.opendaec(daec_path, write=True) as f:
            f.write("x", 1.0)
            f.set_attribute("x", "units", "dollars")
        with de.opendaec(daec_path) as f:
            assert f.get_attribute("x", "units") == "dollars"


class TestObjectMetadata:
    def test_load_object(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        info = memfile.load_object("x")
        assert info.name == "x"
        assert info.obj_class is Class.SCALAR
        assert info.obj_type is Type.FLOAT

    def test_fullpath_and_depth(self, memfile: de.DEFile) -> None:
        memfile.write("a/b/c", 1.0)
        path, depth, created = memfile.get_object_info("a/b/c")
        assert path == "/a/b/c"
        assert depth == 3
        assert created > 0

    def test_catalog_size(self, memfile: de.DEFile) -> None:
        memfile.write("cat", {"a": 1.0, "b": 2.0, "c": 3.0})
        assert memfile.catalog_size("cat") == 3
        assert memfile.catalog_size(de.ROOT_ID) == 1

    def test_find_object_and_fullpath(self, memfile: de.DEFile) -> None:
        memfile.write("a/b", 1.0)
        parent = memfile.find_fullpath("/a")
        assert memfile.find_object(parent, "b") == memfile.find_fullpath("/a/b")

    def test_missing_object_raises(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DEObjectDoesNotExistError):
            memfile.find_fullpath("/nope")

    def test_missing_object_can_be_tolerated(self, memfile: de.DEFile) -> None:
        assert memfile.find_fullpath("/nope", missing_ok=True) is None

    def test_root_resolves(self, memfile: de.DEFile) -> None:
        assert memfile.find_fullpath("/") == de.ROOT_ID


class TestSearching:
    @pytest.fixture
    def populated(self, memfile: de.DEFile) -> de.DEFile:
        memfile.write("alpha", 1.0)
        memfile.write("beta", np.arange(3.0))
        memfile.write("group", {"gamma": 2.0, "delta": {"epsilon": 3.0}})
        return memfile

    def test_list_catalog(self, populated: de.DEFile) -> None:
        names = sorted(o.name for o in populated.list_catalog())
        assert names == ["alpha", "beta", "group"]

    def test_walk_yields_leaf_paths(self, populated: de.DEFile) -> None:
        assert sorted(populated.walk()) == [
            "/alpha",
            "/beta",
            "/group/delta/epsilon",
            "/group/gamma",
        ]

    def test_walk_respects_max_depth(self, populated: de.DEFile) -> None:
        assert sorted(populated.walk(max_depth=1)) == ["/alpha", "/beta"]

    def test_search_by_name_pattern(self, populated: de.DEFile) -> None:
        found = [o.name for o in populated.search_catalog(de.ROOT_ID, "*a")]
        assert sorted(found) == ["alpha", "beta"]

    def test_search_by_class(self, populated: de.DEFile) -> None:
        found = [o.name for o in populated.search_catalog(de.ROOT_ID, obj_class=Class.SCALAR)]
        assert found == ["alpha"]

    def test_search_across_the_whole_file(self, populated: de.DEFile) -> None:
        found = [o.name for o in populated.search_catalog(-1, obj_class=Class.SCALAR)]
        assert sorted(found) == ["alpha", "epsilon", "gamma"]

    def test_search_is_reusable_after_exhaustion(self, populated: de.DEFile) -> None:
        assert list(populated.search_catalog()) != []
        assert list(populated.search_catalog()) != []


class TestNames:
    def test_a_name_with_a_slash_is_rejected(self, memfile: de.DEFile) -> None:
        parent = memfile.makedirs("/cat")
        with pytest.raises(de.DEBadNameError):
            memfile.store_scalar(parent, "a/b", Type.FLOAT, 0, b"\x00" * 8)

    def test_an_empty_name_is_rejected(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DEError):
            memfile.write("", 1.0)

    def test_a_blank_name_is_rejected(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DEBadNameError):
            memfile.store_scalar(de.ROOT_ID, "   ", Type.FLOAT, 0, b"\x00" * 8)

    def test_unicode_names_work(self, memfile: de.DEFile) -> None:
        memfile.write("λ_series", 1.0)
        assert memfile.read("λ_series") == 1.0

    def test_makedirs_is_idempotent(self, memfile: de.DEFile) -> None:
        assert memfile.makedirs("/a/b") == memfile.makedirs("/a/b")

    def test_makedirs_over_a_non_catalog_is_reported(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        with pytest.raises(de.DEExistsError, match="not a catalog"):
            memfile.makedirs("/x/y")


class TestDeletion:
    def test_delete_removes_the_object(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.delete_object("x")
        assert "x" not in memfile

    def test_deleting_the_root_is_refused(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DEError):
            memfile.delete_object(de.ROOT_ID)

    def test_deleting_an_object_removes_its_attributes(self, memfile: de.DEFile) -> None:
        memfile.write("x", 1.0)
        memfile.set_attribute("x", "k", "v")
        memfile.delete_object("x")
        memfile.write("x", 2.0)
        assert memfile.get_attribute("x", "k") is None


class TestUnsupportedValues:
    def test_an_unstorable_object_is_reported_clearly(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DEUnsupportedError, match="Cannot store"):
            memfile.write("x", object())

    def test_a_mixed_object_array_is_reported_clearly(self, memfile: de.DEFile) -> None:
        value = np.array([1, "two", 3.0], dtype=object)
        with pytest.raises(de.DEUnsupportedError, match="not all strings"):
            memfile.write("x", value)

    def test_axis_names_may_not_contain_a_newline(self, memfile: de.DEFile) -> None:
        with pytest.raises(de.DEArgumentError, match="newline"):
            memfile.axis_names(["a\nb", "c"])
