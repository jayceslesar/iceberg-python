# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from pyiceberg.catalog.memory import InMemoryCatalog
from pyiceberg.io import PY_IO_IMPL
from pyiceberg.io.fsspec import FsspecFileIO
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.maintenance import RemoveOrphansResult, _find_orphaned_files, _relative_path
from pyiceberg.types import IntegerType, NestedField, StringType

ARROW_SCHEMA = pa.schema(
    [
        pa.field("city", pa.string(), nullable=False),
        pa.field("inhabitants", pa.int32(), nullable=False),
    ]
)


@pytest.fixture(params=[PyArrowFileIO, FsspecFileIO], ids=["pyarrow", "fsspec"])
def io_impl(request: pytest.FixtureRequest) -> str:
    return f"{request.param.__module__}.{request.param.__name__}"


@pytest.fixture(params=["file://{path}", "{path}"], ids=["file-scheme", "plain-path"])
def catalog(request: pytest.FixtureRequest, tmp_path: Path, io_impl: str) -> InMemoryCatalog:
    warehouse = request.param.format(path=tmp_path.as_posix())
    catalog = InMemoryCatalog("test.in_memory.catalog", warehouse=warehouse, **{PY_IO_IMPL: io_impl})
    catalog.create_namespace("default")
    return catalog


def _create_table_with_snapshots(catalog: InMemoryCatalog, identifier: str) -> Table:
    schema = Schema(
        NestedField(1, "city", StringType(), required=True),
        NestedField(2, "inhabitants", IntegerType(), required=True),
    )
    tbl = catalog.create_table(identifier, schema=schema)
    tbl.append(pa.Table.from_pylist([{"city": "Drachten", "inhabitants": 45019}], schema=ARROW_SCHEMA))
    tbl.append(pa.Table.from_pylist([{"city": "Amsterdam", "inhabitants": 921402}], schema=ARROW_SCHEMA))
    return tbl


def _local_path(location: str) -> Path:
    return Path(location.removeprefix("file://"))


def _write_file(tbl: Table, relative_path: str, age: timedelta = timedelta(0)) -> str:
    location = f"{tbl.location()}/{relative_path}"
    with tbl.io.new_output(location).create() as f:
        f.write(b"orphan")
    mtime = (datetime.now(timezone.utc) - age).timestamp()
    os.utime(_local_path(location), (mtime, mtime))
    return location


def _assert_known_files_exist(tbl: Table) -> None:
    all_known_files = tbl.inspect._all_known_files()
    assert all_known_files["data_files"]
    for files in all_known_files.values():
        for file in files:
            assert _local_path(file).exists(), f"Known file {file} was removed"


def test_remove_orphaned_files(catalog: InMemoryCatalog) -> None:
    identifier = "default.test_remove_orphaned_files"
    tbl = _create_table_with_snapshots(catalog, identifier)

    old_orphans = {
        _write_file(tbl, "data/orphan.parquet", age=timedelta(days=5)),
        _write_file(tbl, "metadata/orphan.avro", age=timedelta(days=5)),
        _write_file(tbl, "nested/directory/orphan.txt", age=timedelta(days=5)),
    }
    fresh_orphan = _write_file(tbl, "data/fresh-orphan.parquet")

    # a dry run reports the orphans without deleting anything
    dry_run = tbl.maintenance.remove_orphaned_files(dry_run=True)
    assert dry_run == RemoveOrphansResult(orphaned_files=old_orphans)
    for orphan in old_orphans | {fresh_orphan}:
        assert _local_path(orphan).exists()

    # by default only files older than three days are removed
    result = tbl.maintenance.remove_orphaned_files()
    assert result == RemoveOrphansResult(orphaned_files=old_orphans, deleted_files=old_orphans)
    for orphan in old_orphans:
        assert not _local_path(orphan).exists()
    assert _local_path(fresh_orphan).exists()

    # a shorter retention also removes the fresh orphan
    result = tbl.maintenance.remove_orphaned_files(older_than=timedelta(0))
    assert result == RemoveOrphansResult(orphaned_files={fresh_orphan}, deleted_files={fresh_orphan})
    assert not _local_path(fresh_orphan).exists()

    # nothing is left to remove and the table is still intact
    assert tbl.maintenance.remove_orphaned_files(older_than=timedelta(0)) == RemoveOrphansResult()
    _assert_known_files_exist(tbl)
    assert catalog.load_table(identifier).scan().to_arrow().num_rows == 2


def test_remove_orphaned_files_keeps_files_of_all_snapshots(catalog: InMemoryCatalog) -> None:
    identifier = "default.test_remove_orphaned_files_keeps_files_of_all_snapshots"
    tbl = _create_table_with_snapshots(catalog, identifier)
    # the overwrite marks the files of the previous snapshots as deleted, but they are still reachable
    tbl.overwrite(pa.Table.from_pylist([{"city": "Groningen", "inhabitants": 238147}], schema=ARROW_SCHEMA))
    snapshots = tbl.snapshots()
    assert len(snapshots) == 4

    rows_by_snapshot = {snapshot.snapshot_id: tbl.scan(snapshot_id=snapshot.snapshot_id).to_arrow() for snapshot in snapshots}
    assert [table.num_rows for table in rows_by_snapshot.values()] == [1, 2, 0, 1]

    # age every file of the table so that the retention period does not protect them
    five_days_ago = (datetime.now(timezone.utc) - timedelta(days=5)).timestamp()
    for file in {file for files in tbl.inspect._all_known_files().values() for file in files}:
        os.utime(_local_path(file), (five_days_ago, five_days_ago))

    assert tbl.maintenance.remove_orphaned_files() == RemoveOrphansResult()
    _assert_known_files_exist(tbl)
    for snapshot_id, rows in rows_by_snapshot.items():
        assert tbl.scan(snapshot_id=snapshot_id).to_arrow() == rows


def test_all_known_files(catalog: InMemoryCatalog) -> None:
    identifier = "default.test_all_known_files"
    tbl = _create_table_with_snapshots(catalog, identifier)
    tbl.overwrite(pa.Table.from_pylist([{"city": "Groningen", "inhabitants": 238147}], schema=ARROW_SCHEMA))
    snapshots = tbl.snapshots()

    all_known_files = tbl.inspect._all_known_files()

    assert set(all_known_files) == {
        "metadata",
        "manifest_lists",
        "manifests",
        "data_files",
        "delete_files",
        "statistics",
        "partition_statistics",
    }
    assert tbl.metadata_location in all_known_files["metadata"]
    assert {entry.metadata_file for entry in tbl.metadata.metadata_log} <= all_known_files["metadata"]
    assert all_known_files["manifest_lists"] == {snapshot.manifest_list for snapshot in snapshots}
    assert all_known_files["manifests"] == set(tbl.inspect.all_manifests()["path"].to_pylist())
    for snapshot in snapshots:
        assert set(tbl.inspect.files(snapshot.snapshot_id)["file_path"].to_pylist()) <= all_known_files["data_files"]
    assert all_known_files["data_files"] == set(tbl.inspect.all_data_files()["file_path"].to_pylist())
    assert all_known_files["delete_files"] == set()
    assert all_known_files["statistics"] == set()
    assert all_known_files["partition_statistics"] == set()
    _assert_known_files_exist(tbl)


def test_remove_orphaned_files_reports_failed_deletions(catalog: InMemoryCatalog) -> None:
    identifier = "default.test_remove_orphaned_files_reports_failed_deletions"
    tbl = _create_table_with_snapshots(catalog, identifier)
    missing_file = f"{tbl.location()}/data/does-not-exist.parquet"

    with patch.object(type(tbl.maintenance), "_orphaned_files", return_value={missing_file}):
        result = tbl.maintenance.remove_orphaned_files()

    assert result == RemoveOrphansResult(orphaned_files={missing_file}, failed_deletions={missing_file})
    _assert_known_files_exist(tbl)


def test_find_orphaned_files_compares_equivalent_locations() -> None:
    as_of = datetime(2025, 1, 1, tzinfo=timezone.utc)
    old = datetime(2024, 1, 1, tzinfo=timezone.utc)
    known_files = {"s3://bucket/table/data/a.parquet", "file:///warehouse/table/data/b.parquet"}

    listed_files = [
        ("s3a://bucket/table/data/a.parquet", old),
        ("s3n://bucket/table/data/a.parquet", old),
        ("/warehouse/table/data/b.parquet", old),
        ("s3://bucket/table/data/c.parquet", old),
    ]

    assert _find_orphaned_files(listed_files, known_files, as_of) == {"s3://bucket/table/data/c.parquet"}


def test_find_orphaned_files_only_returns_files_older_than_as_of() -> None:
    as_of = datetime(2025, 1, 1, tzinfo=timezone.utc)

    listed_files = [
        ("s3://bucket/table/data/old.parquet", datetime(2024, 12, 31, tzinfo=timezone.utc)),
        ("s3://bucket/table/data/old-naive.parquet", datetime(2024, 12, 31)),
        ("s3://bucket/table/data/new.parquet", datetime(2025, 1, 2, tzinfo=timezone.utc)),
        ("s3://bucket/table/data/as-of.parquet", as_of),
        ("s3://bucket/table/data/unknown-mtime.parquet", None),
    ]

    assert _find_orphaned_files(listed_files, set(), as_of) == {
        "s3://bucket/table/data/old.parquet",
        "s3://bucket/table/data/old-naive.parquet",
    }


def test_find_orphaned_files_refuses_prefix_mismatch() -> None:
    as_of = datetime(2025, 1, 1, tzinfo=timezone.utc)
    old = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="different scheme or authority"):
        _find_orphaned_files([("s3://other-bucket/table/data/a.parquet", old)], {"s3://bucket/table/data/a.parquet"}, as_of)

    with pytest.raises(ValueError, match="different scheme or authority"):
        _find_orphaned_files([("gs://bucket/table/data/a.parquet", old)], {"s3://bucket/table/data/a.parquet"}, as_of)


def test_relative_path() -> None:
    assert _relative_path("bucket/table/data/a.parquet", "bucket/table") == "data/a.parquet"
    assert _relative_path("bucket/table/data/a.parquet", "bucket/table/") == "data/a.parquet"
    assert _relative_path("C:/warehouse/table/data/a.parquet", "C:\\warehouse\\table") == "data/a.parquet"

    with pytest.raises(ValueError, match="is not located under"):
        _relative_path("bucket/table-2/data/a.parquet", "bucket/table")
