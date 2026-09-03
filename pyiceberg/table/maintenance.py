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
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pyiceberg.io import _is_local_path
from pyiceberg.utils.concurrent import ExecutorFactory

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pyiceberg.table import Table
    from pyiceberg.table.update.snapshot import ExpireSnapshots

# Schemes that refer to the same storage and are treated as equal when comparing file locations
_EQUIVALENT_SCHEMES = {"": "file", "s3a": "s3", "s3n": "s3"}


@dataclass(frozen=True, kw_only=True)
class RemoveOrphansResult:
    """The outcome of a ``remove_orphaned_files`` run.

    Attributes:
        orphaned_files: Every orphaned file that was found, whether or not it was deleted.
        deleted_files: The orphaned files that were deleted. Empty for a dry run.
        failed_deletions: The orphaned files that could not be deleted. Empty for a dry run.
    """

    orphaned_files: set[str] = field(default_factory=set)
    deleted_files: set[str] = field(default_factory=set)
    failed_deletions: set[str] = field(default_factory=set)


def _comparable_location(location: str) -> tuple[str, str, str]:
    """Split a location into (scheme, authority, path) for comparing files across different spellings of the same URI."""
    if _is_local_path(location):
        return "file", "", location.replace("\\", "/")
    uri = urlparse(location)
    scheme = uri.scheme.lower()
    return _EQUIVALENT_SCHEMES.get(scheme, scheme), uri.netloc, uri.path


def _relative_path(path: str, root: str) -> str:
    """Return the part of ``path`` below ``root``, raising when ``path`` is not under ``root``."""
    normalized_root = root.replace("\\", "/").rstrip("/")
    normalized_path = path.replace("\\", "/")
    if not normalized_path.startswith(f"{normalized_root}/"):
        raise ValueError(f"Listed file {path!r} is not located under {root!r}")
    return normalized_path[len(normalized_root) + 1 :]


def _as_utc(mtime: datetime | None) -> datetime | None:
    if mtime is not None and mtime.tzinfo is None:
        return mtime.replace(tzinfo=timezone.utc)
    return mtime


def _find_orphaned_files(
    listed_files: Iterable[tuple[str, datetime | None]], known_files: Iterable[str], as_of: datetime
) -> set[str]:
    """Return the listed files that are not referenced by the table and were last modified before ``as_of``.

    Locations are compared on their (scheme, authority, path) after mapping equivalent schemes onto each other,
    so ``s3a://bucket/file`` and ``s3://bucket/file`` refer to the same file. A listed file whose path matches a
    known file, but under a different scheme or authority, is refused rather than deleted, since the table would
    be corrupted when the two spellings turn out to point to the same object.

    Args:
        listed_files: The files found in storage, with their last modification time (``None`` when unknown).
        known_files: The files referenced by the table metadata.
        as_of: Only files modified before this moment are considered orphaned.

    Returns:
        The locations of the orphaned files, as they were listed.

    Raises:
        ValueError: When a listed file matches a known file except for its scheme or authority.
    """
    known_prefixes_by_path: dict[str, set[tuple[str, str]]] = {}
    for known_file in known_files:
        scheme, authority, path = _comparable_location(known_file)
        known_prefixes_by_path.setdefault(path, set()).add((scheme, authority))

    orphaned_files: set[str] = set()
    for location, mtime in listed_files:
        scheme, authority, path = _comparable_location(location)
        known_prefixes = known_prefixes_by_path.get(path)
        if known_prefixes is None:
            mtime = _as_utc(mtime)
            if mtime is not None and mtime < as_of:
                orphaned_files.add(location)
        elif (scheme, authority) not in known_prefixes:
            expected = ", ".join(
                sorted(f"{known_scheme}://{known_authority}" for known_scheme, known_authority in known_prefixes)
            )
            raise ValueError(
                f"Unable to determine whether {location} is orphaned: the table references the same path "
                f"under a different scheme or authority ({expected}). Refusing to delete any files."
            )
    return orphaned_files


class MaintenanceTable:
    tbl: Table

    def __init__(self, tbl: Table) -> None:
        self.tbl = tbl

    def _list_files(self, location: str) -> Iterator[tuple[str, datetime | None]]:
        """Recursively list the files under ``location`` together with their last modification time.

        Locations are yielded with the same prefix as ``location``, so that they can be handed back to
        the table's ``FileIO`` and compared with the locations stored in the table metadata.
        """
        from pyiceberg.io.fsspec import FsspecFileIO

        prefix = location.rstrip("/")
        if isinstance(self.tbl.io, FsspecFileIO):
            fs = self.tbl.io._get_fs_from_uri(urlparse(location), location)
            root = fs._strip_protocol(location)
            try:
                for path in fs.find(root):
                    yield f"{prefix}/{_relative_path(path, root)}", fs.modified(path)
            finally:
                # the listing is cached by the filesystem, drop it so that deletions made through other
                # filesystem instances (the executor threads get their own) are visible afterwards
                fs.invalidate_cache()
        else:
            from pyarrow.fs import FileSelector, FileType

            from pyiceberg.io.pyarrow import PyArrowFileIO

            if not isinstance(self.tbl.io, PyArrowFileIO):
                raise ValueError(f"Listing files is not supported for {type(self.tbl.io).__name__}")

            scheme, netloc, root = self.tbl.io.parse_location(location, self.tbl.io.properties)
            fs = self.tbl.io.fs_by_scheme(scheme, netloc)
            for file_info in fs.get_file_info(FileSelector(root, recursive=True)):
                if file_info.type == FileType.File:
                    yield f"{prefix}/{_relative_path(file_info.path, root)}", file_info.mtime

    def _orphaned_files(self, location: str, older_than: timedelta) -> set[str]:
        """Find the files under ``location`` that are not referenced by any metadata of the table.

        Args:
            location: The location to search for orphaned files.
            older_than: Only files that were last modified longer ago than this are considered orphaned.

        Returns:
            The locations of the orphaned files.
        """
        as_of = datetime.now(timezone.utc) - older_than
        known_files = {known_file for files in self.tbl.inspect._all_known_files().values() for known_file in files}
        return _find_orphaned_files(self._list_files(location), known_files, as_of)

    def remove_orphaned_files(self, older_than: timedelta = timedelta(days=3), dry_run: bool = False) -> RemoveOrphansResult:
        """Remove files under the table location that are not referenced by any metadata of the table.

        Orphaned files are files in the table location that are no longer tracked by any snapshot, for example
        because a write failed after the files were written but before they were committed. Files that are
        referenced by any snapshot of the table, the metadata files, the manifest lists, the manifests and the
        statistics files are never removed.

        Args:
            older_than: Only remove files that were last modified longer ago than this. Defaults to three days, so
                that files of a write that is still in progress are not removed.
            dry_run: When True, report the orphaned files without deleting them. Defaults to False.

        Returns:
            The orphaned files that were found, deleted and failed to delete.
        """
        location = self.tbl.location()
        orphaned_files = self._orphaned_files(location, older_than)

        if not orphaned_files:
            logger.info("No orphaned files found at %s", location)
            return RemoveOrphansResult(orphaned_files=orphaned_files)

        if dry_run:
            logger.info("(Dry run) Found %d orphaned files at %s", len(orphaned_files), location)
            return RemoveOrphansResult(orphaned_files=orphaned_files)

        def _delete(file: str) -> tuple[str, bool]:
            try:
                self.tbl.io.delete(file)
                return file, True
            except Exception:
                logger.warning("Failed to delete orphaned file %s", file, exc_info=True)
                return file, False

        executor = ExecutorFactory.get_or_create()
        outcomes = dict(executor.map(_delete, orphaned_files))
        deleted_files = {file for file, deleted in outcomes.items() if deleted}
        failed_deletions = {file for file, deleted in outcomes.items() if not deleted}

        logger.info("Deleted %d orphaned files at %s", len(deleted_files), location)
        if failed_deletions:
            logger.warning("Failed to delete %d orphaned files at %s", len(failed_deletions), location)

        return RemoveOrphansResult(orphaned_files=orphaned_files, deleted_files=deleted_files, failed_deletions=failed_deletions)

    def expire_snapshots(self) -> ExpireSnapshots:
        """Return an ExpireSnapshots builder for snapshot expiration operations.

        Returns:
            ExpireSnapshots builder for configuring and executing snapshot expiration.
        """
        from pyiceberg.table import Transaction
        from pyiceberg.table.update.snapshot import ExpireSnapshots

        return ExpireSnapshots(transaction=Transaction(self.tbl, autocommit=True))
