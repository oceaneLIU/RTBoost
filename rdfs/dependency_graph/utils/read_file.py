from pathlib import Path

import chardet

from rdfs.dependency_graph.models import PathLike

default_encoding = "utf-8"
common_encodings = (
    "utf-8",
    "utf-16",
    "latin-1",
    "ascii",
    "windows-1252",
    "cp1251",
    "cp1253",
    "cp1254",
    "cp1255",
    "cp1256",
    "shift_jis",
    "big5",
    "gb2312",
)


def detect_file_encoding(file_path: Path) -> str:
    """Function to detect encoding"""
    # Read the file as binary data
    raw_data = file_path.read_bytes()
    # Detect encoding
    detected = chardet.detect(raw_data)
    encoding = detected["encoding"]
    return encoding


def read_file_with_encodings(file_path: Path, encodings: tuple[str]) -> tuple[str, str]:
    """Attempt to read a file using various encodings, return content if successful"""
    for encoding in encodings:
        try:
            content = file_path.read_text(encoding=encoding)
            return content, encoding
        except (UnicodeDecodeError, TypeError):
            continue
    raise ValueError(
        f"Could not read file with any of the provided encodings: {encodings}"
    )


def read_file_to_string(file_path: PathLike) -> str:
    """Function to detect encoding and read file to string"""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        content, _ = read_file_with_encodings(file_path, (default_encoding,))
        return content
    except ValueError:
        pass

    try:
        detected_encoding = detect_file_encoding(file_path)
        # Read the file with the detected encoding
        content, _ = read_file_with_encodings(file_path, (detected_encoding,))
        return content
    except ValueError:
        pass

    try:
        content, _ = read_file_with_encodings(file_path, common_encodings)
        return content
    except ValueError:
        raise ValueError(f"Could not read file: {file_path}")
