from pathlib import Path


def resolve_frame_list_csv(
    source: str | Path,
    cruise: str,
) -> Path | None:
    """Resolve one explicit CSV or one cruise frame-list CSV in a directory."""
    source_path = Path(source)
    if source_path.is_file():
        if source_path.suffix.lower() != ".csv":
            raise ValueError(f"Media-list file must be a CSV: {source_path}")
        return source_path

    if not source_path.is_dir():
        return None

    matches = sorted(
        path
        for path in source_path.iterdir()
        if path.is_file()
        and cruise in path.name
        and ("_frame_list_" in path.name or path.name.endswith("_frame_list.csv"))
        and path.suffix.lower() == ".csv"
    )
    if len(matches) > 1:
        match_list = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple frame-list CSVs match cruise {cruise} in {source_path}: "
            f"{match_list}. Pass the exact CSV path."
        )

    return matches[0] if matches else None
