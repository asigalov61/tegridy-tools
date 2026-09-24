"""
robust_file_splitter.py

Robust archival file splitter and reassembler.

Features
--------
* Split files and folders.
* Human-readable headers and footers.
* Split-ID support.
* JSON manifest generation.
* SHA256 verification.
* tqdm progress bars (optional).
* Streaming processing.
* Binary-safe reconstruction.
* Validation of missing/duplicate parts.
* Per-part consistency checks against the manifest.
* Atomic output writes during reassembly.

Project Los Angeles / Tegridy Code 2026
Apache 2.0 / Version 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------------------
# Optional tqdm
# ---------------------------------------------------------------------

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def progress(iterable, enabled=True, **kwargs):
    """Wrap an iterable with tqdm when tqdm is available and enabled."""
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

FORMAT_VERSION = "1.0"

HEADER_BEGIN = "<<<FILE-SPLIT-HEADER-BEGIN>>>"
HEADER_END = "<<<FILE-SPLIT-HEADER-END>>>"

FOOTER_BEGIN = "<<<FILE-SPLIT-FOOTER-BEGIN>>>"
FOOTER_END = "<<<FILE-SPLIT-FOOTER-END>>>"

SEPARATOR = "=" * 80

MIN_DIGITS = 5

# Header blocks are tiny; this is always enough to parse the metadata
# of a well-formed part (a full-read fallback covers exotic cases).
METADATA_PEEK_SIZE = 64 * 1024

# Pre-encoded markers for binary searching.
HEADER_BEGIN_B = HEADER_BEGIN.encode("utf-8")
HEADER_END_B = HEADER_END.encode("utf-8")
FOOTER_BEGIN_B = FOOTER_BEGIN.encode("utf-8")
FOOTER_END_B = FOOTER_END.encode("utf-8")


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class FileSplitError(Exception):
    """Base class for all splitting/reassembly errors."""


class ReassemblyError(FileSplitError):
    """Raised when a split set cannot be reassembled or verified."""


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def sha256_file(
    path,
    buffer_size: int = 1024 * 1024,
) -> str:
    """Streaming SHA256 hex digest of a file."""
    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            block = f.read(buffer_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


# ---------------------------------------------------------------------
# Metadata Creation
# ---------------------------------------------------------------------

def build_header(
    *,
    filename: str,
    file_size: int,
    file_sha256: str,
    split_id: str,
    created_utc: str,
    part_number: int,
    total_parts: int,
    payload_size: int,
) -> bytes:
    """
    Build the header block for one part.

    IMPORTANT: the header ends exactly at HEADER_END with no trailing
    newline, so the payload starts immediately after the marker. The
    reassembler relies on this for binary-safe extraction -- do not
    append separator bytes here.
    """

    text = "\n".join(
        [
            HEADER_BEGIN,
            SEPARATOR,
            "FILE SPLIT CHUNK",
            SEPARATOR,
            f"Format Version : {FORMAT_VERSION}",
            f"Created UTC    : {created_utc}",
            "",
            f"Original File  : {filename}",
            f"Original Size  : {file_size}",
            f"Original SHA256: {file_sha256}",
            "",
            f"Split ID       : {split_id}",
            f"Part Number    : {part_number}",
            f"Parts Total    : {total_parts}",
            f"Payload Bytes  : {payload_size}",
            SEPARATOR,
            HEADER_END,
        ]
    )

    return text.encode("utf-8")


def build_footer(
    *,
    filename: str,
    split_id: str,
    part_number: int,
    total_parts: int,
) -> bytes:
    """
    Build the footer block for one part.

    IMPORTANT: the footer starts exactly at FOOTER_BEGIN with no
    leading newline, so the payload ends immediately before the
    marker -- do not prepend separator bytes here.
    """

    if part_number < total_parts:

        status = [
            "CONTINUES IN NEXT PART",
            f"Next Part      : {part_number + 1}",
        ]

    else:

        status = [
            "END OF FILE",
            "No additional parts follow.",
        ]

    text = "\n".join(
        [
            FOOTER_BEGIN,
            SEPARATOR,
            *status,
            f"Original File  : {filename}",
            f"Split ID       : {split_id}",
            SEPARATOR,
            FOOTER_END,
            "",
        ]
    )

    return text.encode("utf-8")


# ---------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------

def split_file(
    file_path,
    chunk_size,
    output_dir,
    verbose=True,
    show_progress=True,
):
    """
    Split a single file into header/payload/footer part files.

    Returns the path of the JSON manifest, which is written only after
    every part has been created successfully (an existing manifest
    therefore implies a complete split). Empty files are rejected.
    """

    file_path = Path(file_path)
    output_dir = Path(output_dir)

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            "chunk_size must be a positive integer"
        )

    if not file_path.is_file():
        raise FileSplitError(
            f"Not a file: {file_path}"
        )

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise ValueError(
            f"Empty file: {file_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_utc = utc_now()

    file_sha256 = sha256_file(file_path)

    split_id = uuid.uuid4().hex[:8].upper()

    # FIX: integer ceiling division. Float division
    # (math.ceil(file_size / chunk_size)) can miscount very large files.
    total_parts = (file_size + chunk_size - 1) // chunk_size

    digits = max(
        MIN_DIGITS,
        len(str(total_parts)),
    )

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_utc": created_utc,
        "split_id": split_id,
        "original_filename": file_path.name,
        "original_size": file_size,
        "original_sha256": file_sha256,
        "chunk_size": chunk_size,
        "parts": total_parts,
    }

    manifest_path = (
        output_dir
        / f"{file_path.name}.{split_id}.manifest.json"
    )

    if verbose:
        print(
            f"Splitting {file_path.name} "
            f"({file_size} bytes) into {total_parts} part(s)"
        )

    written_parts = []

    try:

        with file_path.open("rb") as src:

            iterator = progress(
                range(1, total_parts + 1),
                enabled=show_progress,
                desc=file_path.name,
                unit="part",
            )

            bytes_read = 0

            for part_number in iterator:

                payload = src.read(chunk_size)

                if not payload:
                    raise FileSplitError(
                        f"File ended unexpectedly during split: "
                        f"{file_path}"
                    )

                bytes_read += len(payload)

                header = build_header(
                    filename=file_path.name,
                    file_size=file_size,
                    file_sha256=file_sha256,
                    split_id=split_id,
                    created_utc=created_utc,
                    part_number=part_number,
                    total_parts=total_parts,
                    payload_size=len(payload),
                )

                footer = build_footer(
                    filename=file_path.name,
                    split_id=split_id,
                    part_number=part_number,
                    total_parts=total_parts,
                )

                # FIX: the old code appended f'{part_number}.txt' to the
                # path, producing malformed names such as
                # 'part-00001-of-000051.txt' (duplicated part number and
                # a garbled total). The extension is now part of the
                # name template itself.
                output_name = (
                    f"{file_path.name}."
                    f"{split_id}."
                    f"part-{part_number:0{digits}d}"
                    f"-of-{total_parts:0{digits}d}"
                    f".txt"
                )

                output_path = output_dir / output_name

                with output_path.open("wb") as out:

                    out.write(header)
                    out.write(payload)
                    out.write(footer)

                written_parts.append(output_path)

        if bytes_read != file_size:
            raise FileSplitError(
                f"File changed during split: "
                f"expected {file_size} bytes, read {bytes_read}"
            )

    except Exception:

        # Never leave orphaned partial parts behind.
        for part in written_parts:

            try:
                part.unlink()
            except OSError:
                pass

        raise

    # FIX: manifest is written last, so it only exists for complete
    # splits (an interrupted split no longer leaves a broken manifest).
    with manifest_path.open("w", encoding="utf-8") as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    if verbose:
        print(
            f"Created {total_parts} part(s) "
            f"+ manifest {manifest_path.name}"
        )

    return manifest_path


def split_folder(
    folder_path,
    chunk_size,
    output_dir,
    verbose=True,
    show_progress=True,
):
    """
    Split every top-level (non-recursive) file in a folder.

    Splitter artifacts (part files / manifests) are skipped, so it is
    safe to split into a folder that already contains previous output.

    Returns the list of manifest paths.
    """

    folder = Path(folder_path)

    if not folder.is_dir():
        raise FileSplitError(
            f"Not a folder: {folder}"
        )

    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and ".part-" not in p.name
        and not p.name.endswith(".manifest.json")
    )

    manifests = []

    for file in files:

        manifests.append(
            split_file(
                file,
                chunk_size,
                output_dir,
                verbose=verbose,
                show_progress=show_progress,
            )
        )

    return manifests


# ---------------------------------------------------------------------
# Metadata Parsing
# ---------------------------------------------------------------------

_PART_RE = re.compile(
    r"Part Number\s*:\s*(\d+)"
)

_TOTAL_RE = re.compile(
    r"Parts Total\s*:\s*(\d+)"
)

_FILE_RE = re.compile(
    r"Original File\s*:\s*(.+)"
)

_SIZE_RE = re.compile(
    r"Original Size\s*:\s*(\d+)"
)

_HASH_RE = re.compile(
    r"Original SHA256\s*:\s*([a-fA-F0-9]+)"
)

_SPLIT_RE = re.compile(
    r"Split ID\s*:\s*(\S+)"
)

_PAYLOAD_RE = re.compile(
    r"Payload Bytes\s*:\s*(\d+)"
)


def require_match(regex, text, field):
    """Search `text` with `regex`, raising if the field is absent."""

    m = regex.search(text)

    if not m:
        raise ReassemblyError(
            f"Missing metadata field: {field}"
        )

    return m


def read_chunk_metadata(
    path,
    peek_size=METADATA_PEEK_SIZE,
):
    """
    Parse the header of a single part file.

    Only the first `peek_size` bytes are read initially (headers are
    tiny); a full read is used only if the header end marker is not
    found within that window.
    """

    path = Path(path)

    with path.open("rb") as f:
        data = f.read(peek_size)

    start = data.find(
        HEADER_BEGIN_B
    )

    if start < 0:
        raise ReassemblyError(
            f"Header begin marker not found: {path.name}"
        )

    end = data.find(
        HEADER_END_B
    )

    if end < 0:

        # Header longer than the peek window; read the whole file.
        data = path.read_bytes()
        start = data.find(HEADER_BEGIN_B)
        end = data.find(HEADER_END_B)

        if end < 0:
            raise ReassemblyError(
                f"Header end marker not found: {path.name}"
            )

    header = data[start:end].decode(
        "utf-8",
        errors="replace"
    )

    return {
        "filename":
            require_match(
                _FILE_RE,
                header,
                "filename"
            ).group(1).strip(),

        "file_size":
            int(
                require_match(
                    _SIZE_RE,
                    header,
                    "size"
                ).group(1)
            ),

        "sha256":
            require_match(
                _HASH_RE,
                header,
                "sha256"
            ).group(1),

        "split_id":
            require_match(
                _SPLIT_RE,
                header,
                "split id"
            ).group(1),

        "part_number":
            int(
                require_match(
                    _PART_RE,
                    header,
                    "part"
                ).group(1)
            ),

        "total_parts":
            int(
                require_match(
                    _TOTAL_RE,
                    header,
                    "total"
                ).group(1)
            ),

        "payload_size":
            int(
                require_match(
                    _PAYLOAD_RE,
                    header,
                    "payload size"
                ).group(1)
            )
    }


# ---------------------------------------------------------------------
# Payload Extraction
# ---------------------------------------------------------------------

def extract_payload(path):
    """
    Extract the raw payload bytes from a single part file.

    Payload boundaries are exactly the markers: the header block ends
    at HEADER_END and the footer block begins at FOOTER_BEGIN, with no
    separator bytes in between, so this slice is binary-safe.
    """

    path = Path(path)

    data = path.read_bytes()

    header_end = data.find(
        HEADER_END_B
    )

    if header_end < 0:
        raise ReassemblyError(
            f"Header end marker missing: {path.name}"
        )

    footer_begin = data.rfind(
        FOOTER_BEGIN_B
    )

    if footer_begin < 0:
        raise ReassemblyError(
            f"Footer begin marker missing: {path.name}"
        )

    payload_start = (
        header_end
        + len(HEADER_END_B)
    )

    if footer_begin < payload_start:
        raise ReassemblyError(
            f"Malformed part (footer precedes payload): {path.name}"
        )

    if data.find(FOOTER_END_B, footer_begin) < 0:
        raise ReassemblyError(
            f"Footer end marker missing: {path.name}"
        )

    return data[
        payload_start:
        footer_begin
    ]


# ---------------------------------------------------------------------
# Reassembler
# ---------------------------------------------------------------------

def reassemble_split_set(
    chunk_folder,
    manifest_file,
    output_dir,
    verbose=True,
    show_progress=True,
):
    """
    Reassemble a single split set identified by its manifest.

    The result is written to a temporary file and only moved into its
    final location after size and SHA256 verification succeed.
    """

    chunk_folder = Path(chunk_folder)
    manifest_file = Path(manifest_file)
    output_dir = Path(output_dir)

    if not chunk_folder.is_dir():
        raise ReassemblyError(
            f"Not a folder: {chunk_folder}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_file.open("r", encoding="utf-8") as f:

        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ReassemblyError(
            f"Manifest is not a JSON object: {manifest_file.name}"
        )

    if manifest.get("format_version") not in (None, FORMAT_VERSION):

        if verbose:
            print(
                f"Warning: {manifest_file.name} has format version "
                f"{manifest.get('format_version')!r}, "
                f"expected {FORMAT_VERSION!r}"
            )

    required = [
        "split_id",
        "parts",
        "original_filename",
        "original_size",
        "original_sha256",
    ]

    for key in required:

        if key not in manifest:
            raise ReassemblyError(
                f"Manifest missing {key}: {manifest_file.name}"
            )

    split_id = str(manifest["split_id"]).strip()

    if not split_id:
        raise ReassemblyError(
            f"Manifest has an empty split_id: {manifest_file.name}"
        )

    total_parts = manifest["parts"]

    if not isinstance(total_parts, int) or total_parts < 1:
        raise ReassemblyError(
            f"Invalid parts count in manifest: {total_parts!r}"
        )

    original_filename = manifest["original_filename"]

    if (
        not isinstance(original_filename, str)
        or not original_filename.strip()
    ):
        raise ReassemblyError(
            f"Manifest has an invalid original_filename: "
            f"{original_filename!r}"
        )

    # FIX: never trust the manifest when constructing the output path
    # (prevents path traversal via a crafted manifest).
    safe_name = Path(original_filename.replace("\\", "/")).name

    if safe_name in ("", ".", ".."):
        raise ReassemblyError(
            f"Manifest has an unsafe original_filename: "
            f"{original_filename!r}"
        )

    try:
        original_size = int(manifest["original_size"])
    except (TypeError, ValueError):
        raise ReassemblyError(
            f"Manifest has an invalid original_size: "
            f"{manifest['original_size']!r}"
        )

    original_sha256 = str(manifest["original_sha256"]).strip().lower()

    if not re.fullmatch(r"[0-9a-f]{64}", original_sha256):
        raise ReassemblyError(
            f"Manifest has an invalid original_sha256: {original_sha256!r}"
        )

    output_file = output_dir / safe_name
    tmp_file = output_file.with_name(output_file.name + ".tmp")
    tmp_name = tmp_file.name

    # FIX: only consider files belonging to this split set. The old
    # '.part-' substring filter also matched directories and unrelated
    # files (crashing on read). This also skips the set's own previous
    # output/temp file when reassembling in place.
    part_token = f".{split_id}.part-"

    parts = []

    for file in chunk_folder.iterdir():

        if not file.is_file():
            continue

        if file.name in (safe_name, tmp_name):
            continue

        if part_token not in file.name:
            continue

        meta = read_chunk_metadata(
            file
        )

        if meta["split_id"] != split_id:
            continue

        # FIX: cross-check every part against the manifest.
        if meta["filename"] != original_filename:
            raise ReassemblyError(
                f"Filename mismatch in {file.name}: "
                f"expected {original_filename!r}, "
                f"found {meta['filename']!r}"
            )

        if meta["file_size"] != original_size:
            raise ReassemblyError(
                f"Original size mismatch in {file.name}: "
                f"expected {original_size}, "
                f"found {meta['file_size']}"
            )

        if meta["sha256"].lower() != original_sha256:
            raise ReassemblyError(
                f"SHA256 mismatch in {file.name}"
            )

        if meta["total_parts"] != total_parts:
            raise ReassemblyError(
                f"Parts total mismatch in {file.name}: "
                f"expected {total_parts}, "
                f"found {meta['total_parts']}"
            )

        parts.append(
            (file, meta)
        )

    if not parts:
        raise ReassemblyError(
            f"No parts found for split ID {split_id} in {chunk_folder}"
        )

    found = sorted(
        p[1]["part_number"]
        for p in parts
    )

    expected = list(
        range(
            1,
            total_parts + 1
        )
    )

    if found != expected:

        missing = sorted(set(expected).difference(found))

        raise ReassemblyError(
            f"Missing or duplicated parts for split ID {split_id}.\n"
            f"Missing part numbers: {missing}\n"
            f"Expected: {expected}\n"
            f"Found: {found}"
        )

    parts.sort(
        key=lambda x:
        x[1]["part_number"]
    )

    try:

        with tmp_file.open("wb") as out:

            if verbose:
                print(
                    f"Reassembling {len(parts)} part(s) "
                    f"-> {output_file.name}"
                )

            iterator = progress(
                parts,
                enabled=show_progress,
                desc="Reassembling",
                unit="part",
            )

            for part_file, meta in iterator:

                payload = extract_payload(
                    part_file
                )

                if len(payload) != meta["payload_size"]:
                    raise ReassemblyError(
                        f"Payload size mismatch in {part_file.name}: "
                        f"header declares {meta['payload_size']} bytes, "
                        f"extracted {len(payload)}"
                    )

                out.write(payload)

        size = tmp_file.stat().st_size

        if size != original_size:
            raise ReassemblyError(
                f"Size verification failed for {output_file.name}: "
                f"expected {original_size}, got {size}"
            )

        sha = sha256_file(tmp_file)

        if sha.lower() != original_sha256:
            raise ReassemblyError(
                f"SHA256 verification failed for {output_file.name}"
            )

        # FIX: publish the output only after every check has passed, so
        # a failed reassembly never leaves a partial/corrupt file.
        tmp_file.replace(output_file)

    except Exception:

        try:
            tmp_file.unlink()
        except OSError:
            pass

        raise

    if verbose:
        print(
            f"Verified: "
            f"{output_file.name} "
            f"({size} bytes, SHA256 OK)"
        )

    return output_file


def reassemble_all(
    chunk_folder,
    output_dir,
    verbose=True,
    show_progress=True,
):
    """
    Reassemble every split set found in `chunk_folder`.

    A manifest that fails is reported and skipped, so one broken set
    does not abort the whole batch (failures are always printed).
    Returns the list of restored files.
    """

    chunk_folder = Path(chunk_folder)

    if not chunk_folder.is_dir():
        raise ReassemblyError(
            f"Not a folder: {chunk_folder}"
        )

    manifests = sorted(
        chunk_folder.glob(
            "*.manifest.json"
        )
    )

    if not manifests:
        raise ReassemblyError(
            f"No manifest files found in {chunk_folder}"
        )

    restored = []

    for manifest in manifests:

        try:

            restored.append(
                reassemble_split_set(
                    chunk_folder,
                    manifest,
                    output_dir,
                    verbose=verbose,
                    show_progress=show_progress,
                )
            )

        except (
            FileSplitError,
            json.JSONDecodeError,
            OSError,
            ValueError,
            KeyError,
        ) as exc:

            print(
                f"FAILED: {manifest.name}: {exc}"
            )

    return restored


if __name__ == "__main__":

    # Example:
    #
    # split_folder(
    #     folder_path="input",
    #     chunk_size=100 * 1024 * 1024,
    #     output_dir="chunks",
    #     verbose=True,
    #     show_progress=True,
    # )
    #
    # restored = reassemble_all(
    #     chunk_folder="chunks",
    #     output_dir="restored",
    #     verbose=True,
    #     show_progress=True,
    # )

    pass