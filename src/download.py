"""Download MoneyPuck's published shot archives, validate, and cache them.

MoneyPuck data: https://moneypuck.com/data.htm (non-commercial use; attribution required).
Only published season links are used. No raw NHL scraping or xG substitution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import zipfile

from src.config import add_config_arguments, config_from_args
from src.paths import DATA_DIR

CATALOG_URL = "https://moneypuck.com/data.htm"
MAX_ARCHIVE = 300 * 1024**2
MAX_CSV = 1024**3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def fetch(url, destination, max_bytes=MAX_ARCHIVE, attempts=3):
    """Stream with bounded retries. A failed request never installs a data file."""
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "hockeybayes/1.0 (research data downloader)"})
            with urlopen(request, timeout=60) as response, Path(destination).open("wb") as stream:
                size = 0
                while True:
                    chunk = response.read(1024**2)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"Download exceeds size limit: {url}")
                    stream.write(chunk)
            return
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise RuntimeError(f"Download failed (HTTP {exc.code}): {url}") from exc
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Download failed: {url}: {exc}") from exc
        time.sleep(2**attempt)


class SeasonLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        url = urljoin(CATALOG_URL, href)
        parsed = urlparse(url)
        match = re.fullmatch(r"shots_(\d{4})\.zip", Path(parsed.path).name)
        if (match and parsed.scheme == "https"
                and parsed.hostname in {"peter-tanner.com", "www.peter-tanner.com", "moneypuck.com", "www.moneypuck.com"}):
            self.links[int(match.group(1))] = url


def catalog_links():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "catalog.html"
        fetch(CATALOG_URL, path, max_bytes=5 * 1024**2)
        parser = SeasonLinks()
        parser.feed(path.read_text())
    if not parser.links:
        raise ValueError("No season archives found on MoneyPuck's data page; its format may have changed")
    return parser.links


def validate_csv(path, year):
    import pandas as pd
    from src.analysis import filter_shots
    df = pd.read_csv(path)
    extra = {"shotDistance", "shotType", "xCordAdjusted", "yCordAdjusted", "homeSkatersOnIce", "awaySkatersOnIce"}
    missing = extra - set(df.columns)
    if missing:
        raise ValueError(f"Missing analysis fields: {sorted(missing)}")
    selected = filter_shots(df, year)
    if selected.empty:
        raise ValueError(f"No qualifying shots in season {year}")
    return {"rows": len(df), "qualifying_shots": len(selected), "games": int(selected.game_key.nunique())}


def ensure_season(year, data_dir, links=None, refresh=False):
    """Install only a validated CSV; preserve old snapshots by content hash."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"shots_{year}.csv"
    sidecar = target.with_suffix(".json")
    if target.exists() and not refresh:
        stats = validate_csv(target, year)
        digest = sha256(target)
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if metadata.get("sha256") != digest:
            metadata = {"source": "local file", "downloaded_at_utc": None}
        metadata.update(year=year, sha256=digest, bytes=target.stat().st_size, **stats)
        metadata["validated_at_utc"] = utc_now()
        atomic_json(sidecar, metadata)
        return metadata
    links = catalog_links() if links is None else links
    if year not in links:
        raise ValueError(f"MoneyPuck has no published archive for {year}; choose another season or supply its CSV locally")
    with tempfile.TemporaryDirectory(dir=data_dir, prefix=".download-") as temp:
        archive = Path(temp) / "shots.zip"
        candidate = Path(temp) / target.name
        fetch(links[year], archive)
        with zipfile.ZipFile(archive) as bundle:
            members = [i for i in bundle.infolist() if i.filename == target.name]
            if len(members) != 1 or members[0].file_size > MAX_CSV:
                raise ValueError(f"Archive must contain exactly one root-level {target.name} within the size limit")
            # Copy only the expected member; never extract arbitrary archive paths.
            with bundle.open(members[0]) as source, candidate.open("wb") as dest:
                shutil.copyfileobj(source, dest)
        stats = validate_csv(candidate, year)
        digest = sha256(candidate)
        metadata = {"year": year, "source": links[year], "downloaded_at_utc": utc_now(),
                    "sha256": digest, "bytes": candidate.stat().st_size, **stats}
        if target.exists() and sha256(target) != digest:
            previous = sha256(target)
            snapshots = data_dir / "snapshots" / str(year)
            snapshots.mkdir(parents=True, exist_ok=True)
            saved = snapshots / f"{previous}.csv"
            if not saved.exists():
                shutil.copy2(target, saved)
            if sidecar.exists():
                shutil.copy2(sidecar, saved.with_suffix(".json"))
        candidate.replace(target)
        atomic_json(sidecar, metadata)
    return metadata


def ensure_data(config, data_dir, refresh_evaluation=False, offline=False):
    needed = config.seasons(data_dir)
    refresh = refresh_evaluation or config.evaluation_status == "season-to-date"
    missing = [p for p in needed if not p.exists()]
    if offline and missing:
        raise FileNotFoundError("Offline mode: missing " + ", ".join(str(p) for p in missing))
    links = None
    if not offline and (missing or refresh):
        links = catalog_links()
    records = []
    for path, meta in needed.items():
        print(f"Checking {path.name}...", flush=True)
        records.append(ensure_season(meta["year"], data_dir, links,
                                    refresh=not offline and refresh and meta["role"] == "validate"))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--refresh-evaluation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        config = config_from_args(args)
        ensure_data(config, args.data_dir, args.refresh_evaluation, args.offline)
    except (ValueError, TypeError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
