# -*- coding: utf-8 -*-
"""Zarr v3 store adapter for reading IMS files."""

import asyncio
import itertools
import os
import sys
import logging
from collections.abc import AsyncIterator, Iterable

import numpy as np
import zarr
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

import imaris_ims_file_reader as ims

logger = logging.getLogger(__name__)


class ims_zarr_store(Store):
    """Zarr v3 store adapter for reading IMS files."""

    PARTIAL_READ_BATCH_SIZE: int = 64

    __hash__ = None

    supports_writes: bool = False
    supports_deletes: bool = False
    supports_listing: bool = True

    def __init__(
        self,
        ims_file,
        ResolutionLevelLock=0,
        writeable=False,
        normalize_keys=True,
        verbose=False,
        mode="r",
    ):
        super().__init__(read_only=True)

        assert os.path.splitext(ims_file)[-1].lower() == ".ims"

        self.path = ims_file
        self.ResolutionLevelLock = ResolutionLevelLock
        self.normalize_keys = normalize_keys
        self.verbose = verbose
        self.writeable = writeable

        self.ims = self.open_ims()
        self.ResolutionLevels = self.ims.ResolutionLevels
        self.TimePoints = self.ims.TimePoints
        self.Channels = self.ims.Channels
        self.chunks = self.ims.chunks
        self.shape = self.ims.shape
        try:
            self.dtype = np.dtype(self.ims.dtype)
        except TypeError as exc:
            raise TypeError(f"Unsupported IMS dtype: {self.ims.dtype!r}") from exc
        self.ndim = self.ims.ndim

        self._zarr_json = self._build_zarr_json()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ims_zarr_store) and self.path == other.path and self.ResolutionLevelLock == other.ResolutionLevelLock

    def open_ims(self):
        return ims.ims(
            self.path,
            ResolutionLevelLock=self.ResolutionLevelLock,
            write=self.writeable,
            squeeze_output=False,
        )

    def _normalize_key(self, key: str) -> str:
        return key.lower() if self.normalize_keys else key

    def _build_zarr_json(self) -> bytes:
        metadata_store: dict[str, Buffer] = {}
        if self.dtype.byteorder in ("<", "|"):
            endian = "little"
        elif self.dtype.byteorder == ">":
            endian = "big"
        elif self.dtype.byteorder == "=":
            endian = sys.byteorder
        else:
            raise ValueError(f"Unsupported dtype byteorder: {self.dtype.byteorder!r}")

        zarr.open_array(
            store=metadata_store,
            mode="w",
            zarr_format=3,
            shape=self.shape,
            chunks=self.chunks,
            dtype=self.dtype,
            fill_value=0,
            chunk_key_encoding=("v2", "."),
            codecs=[{"name": "bytes", "configuration": {"endian": endian}}],
        )

        return metadata_store["zarr.json"].to_bytes()

    @staticmethod
    def _apply_byte_range(data: bytes, byte_range: ByteRequest | None) -> bytes:
        if byte_range is None:
            return data
        if isinstance(byte_range, RangeByteRequest):
            return data[byte_range.start : byte_range.end]
        if isinstance(byte_range, OffsetByteRequest):
            return data[byte_range.offset :]
        if isinstance(byte_range, SuffixByteRequest):
            return data[-byte_range.suffix :]
        return data

    def _chunk_index_from_key(self, key: str) -> list[tuple[int, int]]:
        try:
            key_split = [int(x) for x in key.split(".")]
        except ValueError as exc:
            raise ValueError(f"Invalid chunk key format: {key!r}; expected dot-separated integer indices") from exc
        index = []
        for axis, chunk_idx in enumerate(key_split):
            start = self.chunks[axis] * chunk_idx
            stop = min(start + self.chunks[axis], self.shape[axis])
            index.append((start, stop))
        return index

    def _chunk_bytes(self, index: list[tuple[int, int]]) -> bytes:
        array = self.ims[
            self.ResolutionLevelLock,
            index[0][0] : index[0][1],
            index[1][0] : index[1][1],
            index[2][0] : index[2][1],
            index[3][0] : index[3][1],
            index[4][0] : index[4][1],
        ]

        if tuple(array.shape) != tuple(self.chunks):
            canvas = np.zeros(self.chunks, dtype=array.dtype)
            canvas[
                0 : array.shape[0],
                0 : array.shape[1],
                0 : array.shape[2],
                0 : array.shape[3],
                0 : array.shape[4],
            ] = array
            array = canvas

        return np.ascontiguousarray(array).tobytes(order="C")

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if prototype is None:
            prototype = default_buffer_prototype()

        normalized_key = self._normalize_key(key)

        if normalized_key == "zarr.json":
            return prototype.buffer.from_bytes(
                self._apply_byte_range(self._zarr_json, byte_range)
            )

        try:
            chunk_index = self._chunk_index_from_key(normalized_key)
        except (ValueError, IndexError):
            logger.debug("Invalid chunk key requested: %s", key)
            return None

        data = self._chunk_bytes(chunk_index)
        return prototype.buffer.from_bytes(self._apply_byte_range(data, byte_range))

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        results: list[Buffer | None] = []
        key_ranges_iter = iter(key_ranges)
        batch_size = self.PARTIAL_READ_BATCH_SIZE

        while True:
            batch = list(itertools.islice(key_ranges_iter, batch_size))
            if not batch:
                break
            tasks = [
                self.get(key, prototype=prototype, byte_range=byte_range)
                for key, byte_range in batch
            ]
            results.extend(await asyncio.gather(*tasks))

        return results

    async def exists(self, key: str) -> bool:
        normalized_key = self._normalize_key(key)
        if normalized_key == "zarr.json":
            return True
        try:
            _ = self._chunk_index_from_key(normalized_key)
            return True
        except (ValueError, IndexError):
            return False

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()

    async def delete(self, key: str) -> None:
        self._check_writable()

    async def list(self) -> AsyncIterator[str]:
        yield "zarr.json"
        for key in self._iter_chunk_keys():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        if prefix == "":
            yield "zarr.json"
            return

        async for key in self.list_prefix(prefix + "/"):
            remainder = key[len(prefix) + 1 :]
            if remainder:
                yield remainder.split("/", 1)[0]

    def _iter_chunk_keys(self):
        """Yield all chunk keys for the current IMS array using dot-separated indices."""
        chunk_num = []
        for axis in range(5):
            count = self.shape[axis] // self.chunks[axis]
            if self.shape[axis] % self.chunks[axis] != 0:
                count += 1
            chunk_num.append(count)

        for t, c, z, y, x in itertools.product(
            range(chunk_num[0]),
            range(chunk_num[1]),
            range(chunk_num[2]),
            range(chunk_num[3]),
            range(chunk_num[4]),
        ):
            yield f"{t}.{c}.{z}.{y}.{x}"
