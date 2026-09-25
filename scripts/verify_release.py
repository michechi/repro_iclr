#!/usr/bin/env python3
"""Read-only checks for an anonymous release; uses only Python's standard library.

This is a structural check, not proof of anonymity or a substitute for review of
figures and clinical aggregates. Findings contain paths and issue classes only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zlib


TEXT_SUFFIXES = {
    ".py", ".md", ".rst", ".txt", ".tex", ".bib", ".sty", ".cls", ".bst",
    ".sh", ".slurm", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".csv",
    ".svg", ".def", ".example", ".sha256", "",
}
FORBIDDEN_DIRS = {".git", ".venv", "venv", ".deps", "__pycache__", ".cache", "node_modules", ".ssh"}
FORBIDDEN_SUFFIXES = {
    ".pyc", ".pyo", ".pkl", ".pickle", ".pt", ".pth", ".bin", ".safetensors",
    ".npy", ".npz", ".joblib", ".ubj", ".so", ".log", ".pid", ".out", ".err",
    ".aux", ".fls", ".fdb_latexmk", ".zip", ".tar", ".tgz", ".gz", ".ipynb",
}
PATTERNS = {
    "private_absolute_path": re.compile(r"/(?:root|home|Users|cluster|scratch|dev/shm|fp/projects)/[\w.-]+"),
    "email_address": re.compile(r"(?<![\w.+-])[\w.+-]{1,90}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
    "credential_shape": re.compile(r"\b(?:hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{30,}|(?:ghp|github_pat)_[A-Za-z0-9_]{30,})\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "runtime_process_identity": re.compile(r"\b(?:supervisor\s+|trainer\s+)?PID\s*[:=]?\s*\**\d{2,}\b", re.I),
}
AUTHOR = re.compile(r"^\s*(?:#\s*)?(?:Author|Created by|Maintainer)\s*:\s*(.+)$", re.I)
RESTRICTED_KEYS = {
    "subject_id", "hadm_id", "stay_id", "patient_id", "patient_ids", "subject_ids",
    "sample_indices", "patient_indices", "row_indices", "predictions", "probabilities",
    "labels", "input_ids", "sequences", "patient_records", "per_patient",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_clinical_output(rel: str) -> bool:
    if not (rel.startswith("results/") or rel.startswith("paper/figures/")):
        return False
    return bool(re.search(
        r"(?:/[^/]*_ckd_(?:ordered|shuffled)(?:/|\.)|/clinical_attention/|"
        r"/ckd_paired_comparison/|/qwen[\w.]*_ckd_\d|/attention_ckd_)", rel
    ))


def walk_values(value, trail=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item, trail + "/" + key
            yield from walk_values(item, trail + "/" + key)
    elif isinstance(value, list):
        for item in value:
            yield from walk_values(item, trail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, help="SHA256 manifest, relative to the release root")
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--identity-file", type=Path,
                        help="Optional private line-separated identity denylist, kept outside the release")
    args = parser.parse_args()
    root = args.root.resolve()
    findings: list[dict] = []
    files: list[Path] = []
    identities = []

    def issue(path: Path | str, kind: str, line: int | None = None):
        rel = path.as_posix() if isinstance(path, Path) else path
        item = {"path": rel, "issue": kind}
        if line is not None:
            item["line"] = line
        if item not in findings:
            findings.append(item)

    if args.identity_file:
        if args.identity_file.resolve().is_relative_to(root):
            parser.error("The private identity file must remain outside the release")
        identities = [s.strip().casefold() for s in args.identity_file.read_text().splitlines() if s.strip()]

    def check_text(text: str, rel: Path, line: int | None = None):
        for kind, pattern in PATTERNS.items():
            if kind == "email_address" and "@" not in text:
                continue
            if kind == "private_absolute_path" and "/" not in text:
                continue
            if kind == "credential_shape" and not any(marker in text for marker in ("hf_", "sk-", "ghp_", "github_pat_")):
                continue
            if kind == "private_key" and "BEGIN" not in text:
                continue
            if kind == "runtime_process_identity" and "pid" not in text.lower():
                continue
            if kind == "runtime_process_identity" and "tests" in rel.parts:
                continue  # Process IDs in these unit tests are synthetic fixtures.
            if pattern.search(text):
                issue(rel, kind, line)
        if any(identity in text.casefold() for identity in identities):
            issue(rel, "private_identity_match", line)

    for current, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(current) / name
            rel = path.relative_to(root)
            if rel.parts == (".git",):
                continue  # the release repository's own metadata; its history starts at the anonymous release commit
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                issue(rel, "symlink")
            elif not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                issue(rel, "non_regular_artifact")
            if any(part in FORBIDDEN_DIRS for part in rel.parts):
                issue(rel, "forbidden_directory")
        dirs[:] = [d for d in dirs if d not in FORBIDDEN_DIRS and not (Path(current) / d).is_symlink()]
        for name in names:
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            files.append(path)
            rel = path.relative_to(root)
            suffix = path.suffix.lower()
            if (suffix in FORBIDDEN_SUFFIXES or name in {".DS_Store", "Thumbs.db"}
                    or (name.startswith(".env") and name != ".env.example")):
                issue(rel, "forbidden_artifact")
            if re.search(r"(?:^|/)(?:raw|processed|model_cache|checkpoints)(?:/|$)", rel.as_posix()):
                issue(rel, "restricted_data_or_weights_directory")
            if suffix == ".parquet" and not rel.as_posix().startswith("data/simulation/"):
                issue(rel, "unreviewed_columnar_data")
            if any(identity in rel.as_posix().casefold() for identity in identities):
                issue(rel, "private_identity_in_filename")
            if suffix in TEXT_SUFFIXES or name.startswith(".env"):
                try:
                    with path.open(encoding="utf-8") as handle:
                        if suffix == ".csv":
                            overlap = ""
                            for block in iter(lambda: handle.read(1024 * 1024), ""):
                                check_text(overlap + block, rel)
                                overlap = block[-512:]
                        else:
                            for number, line in enumerate(handle, 1):
                                check_text(line, rel, number)
                                match = AUTHOR.match(line)
                                if match and not match.group(1).strip().casefold().startswith("anonymous"):
                                    issue(rel, "nonanonymous_author_label", number)
                except UnicodeDecodeError:
                    issue(rel, "unexpected_non_utf8_text")
            if is_clinical_output(rel.as_posix()):
                if suffix == ".json":
                    try:
                        obj = json.loads(path.read_text())
                        for key, value, trail in walk_values(obj):
                            if key.casefold() in RESTRICTED_KEYS:
                                issue(rel, "clinical_patient_level_field")
                            aggregate_sources = key == "sources" and all(
                                isinstance(item, dict) and set(item) <= {"condition", "directory", "model"}
                                and all(isinstance(v, str) for v in item.values()) for item in value
                            ) if isinstance(value, list) else False
                            if isinstance(value, list) and key not in {"physical_layer_ids", "layer_ids", "missing_linear_attention_layers"} and not aggregate_sources:
                                issue(rel, "clinical_list_requires_review")
                    except (ValueError, UnicodeDecodeError):
                        issue(rel, "invalid_clinical_json")
                if suffix == ".csv":
                    with path.open(newline="", encoding="utf-8") as handle:
                        columns = next(csv.reader(handle), [])
                    if any(c.casefold() in RESTRICTED_KEYS for c in columns):
                        issue(rel, "clinical_patient_level_column")
                    if set(columns) & {"sequence", "label", "probability", "row_id", "patient_index"}:
                        issue(rel, "clinical_record_column")
            if suffix == ".png":
                data = path.read_bytes()
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    issue(rel, "invalid_png")
                    continue
                offset = 8
                while offset + 12 <= len(data):
                    length = struct.unpack(">I", data[offset:offset + 4])[0]
                    kind = data[offset + 4:offset + 8]
                    value = data[offset + 8:offset + 8 + length]
                    if offset + length + 12 > len(data):
                        issue(rel, "invalid_png_chunk")
                        break
                    if kind == b"eXIf":
                        issue(rel, "image_exif_requires_review")
                    if kind in {b"tEXt", b"zTXt", b"iTXt"}:
                        try:
                            key, content = value.split(b"\0", 1)
                            if kind == b"zTXt":
                                content = zlib.decompress(content[1:])
                            elif kind == b"iTXt":
                                compressed = content[0]
                                content = content[2:].split(b"\0", 2)[2]
                                if compressed:
                                    content = zlib.decompress(content)
                            metadata_text = content.decode("utf-8", errors="replace")
                            check_text(metadata_text, rel)
                            if key.lower() not in {b"software", b"creator"}:
                                issue(rel, "image_text_metadata_requires_review")
                            if key.lower() == b"creator" and not metadata_text.casefold().startswith("anonymous"):
                                issue(rel, "image_creator_requires_review")
                        except (ValueError, IndexError, zlib.error):
                            issue(rel, "invalid_png_metadata")
                    offset += length + 12
            if suffix == ".pdf":
                data = path.read_bytes()
                for match in re.finditer(rb"/(Author|Creator|Title|Subject|Keywords)\s*\(((?:\\.|[^)])*)\)", data):
                    key, content = match.groups()
                    value = content.decode("latin1").strip()
                    check_text(value, rel)
                    if key == b"Author" and value and not value.casefold().startswith("anonymous"):
                        issue(rel, "pdf_author_requires_review")
                if b"xmpmeta" in data:
                    issue(rel, "pdf_xmp_requires_review")

    candidates = [root / "MANIFEST.sha256", root / "release/MANIFEST.sha256"]
    manifest = (root / args.manifest if args.manifest and not args.manifest.is_absolute()
                else args.manifest) or next((p for p in candidates if p.is_file()), None)
    manifest_checked = 0
    if manifest is None or not manifest.is_file():
        if args.require_manifest:
            issue("MANIFEST.sha256", "missing_manifest")
    else:
        expected: set[str] = set()
        for number, line in enumerate(manifest.read_text().splitlines(), 1):
            if not line.strip():
                continue
            match = re.fullmatch(r"([a-fA-F0-9]{64}) [ *](.+)", line)
            if not match:
                issue("MANIFEST.sha256", "invalid_manifest_line", number)
                continue
            wanted, name = match.groups()
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                issue("MANIFEST.sha256", "unsafe_manifest_path", number)
                continue
            canonical = relative.as_posix()
            if canonical in expected:
                issue("MANIFEST.sha256", "duplicate_manifest_entry", number)
                continue
            expected.add(canonical)
            path = root / canonical
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                issue(canonical, "unsafe_manifest_target")
            elif not path.is_file():
                issue(canonical, "manifest_file_missing")
            elif digest(path) != wanted.lower():
                issue(canonical, "manifest_hash_mismatch")
            manifest_checked += 1
        actual = {p.relative_to(root).as_posix() for p in files if p != manifest}
        for name in sorted(actual - expected):
            issue(name, "unmanifested_file")
        for name in sorted(expected - actual):
            if root / name == manifest:
                issue(name, "manifest_must_not_hash_itself")

    print(json.dumps({
        "status": "failed" if findings else "passed",
        "files_checked": len(files),
        "manifest_checked": manifest_checked,
        "manifest_status": "checked" if manifest_checked else "absent_or_empty",
        "issues": findings,
        "scope": "Structural release checks; scientific provenance and figure contents require human review.",
    }, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
