"""Download and prepare the original public nighttime-lights replication data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


HERE = Path(__file__).resolve().parent
DATA_URL = (
    "https://www.econometricsociety.org/publications/econometrica/2013/01/01/"
    "pre%E2%80%90colonial-ethnic-institutions-and-contemporary-african/"
    "supp/9613_data_and_programs_0.zip"
)
ARCHIVE_SHA256 = "7ea2271378956abff4fa940a21d1e9f5e34f7485bee731023f1fef1e3213058d"
DATA_MEMBER = "4web/ethnicity-country-data-final.dta"
MEMBER_SHA256 = "00d1e5aef7d83fea297b88f63fcbdc98336449689aedd0caa02367adc1471ad7"
NUMERIC_FIELDS = (
    "mean_lights0708", "centr_tribe", "lnpdmean00s", "capdistance1",
    "seadist1", "borderdist1", "lnwaterkm", "lnkm2split", "mean_elev",
    "mean_suit", "malariasuit", "petroleum", "diamondd",
)


def prepare(archive: bytes, output: Path) -> Path:
    """Reproduce the used variables, precision, country codes and source order."""
    import pandas as pd

    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("The source archive differs from the version used for replication")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        data = bundle.read(DATA_MEMBER)
    if hashlib.sha256(data).hexdigest() != MEMBER_SHA256:
        raise ValueError("The original Stata data file differs from the expected version")
    frame = pd.read_stata(io.BytesIO(data), convert_categoricals=False)
    order = json.loads((HERE / "row_order.json").read_text())
    ids = frame["id_split"].astype(int)
    if len(frame) != 683 or ids.duplicated().any() or set(ids) != set(order):
        raise ValueError("The original data identifiers do not match the replication sample")
    frame = frame.assign(id_split=ids).set_index("id_split").loc[order]
    countries = frame["wbcode"].astype(str)
    country_codes = {country: i + 1 for i, country in enumerate(sorted(countries.unique()))}
    if len(country_codes) != 49:
        raise ValueError("Expected 49 original country codes before singleton exclusion")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("id_split", *NUMERIC_FIELDS, "wbcode", "cluster"))
    for identifier, row in frame.iterrows():
        # The historical Stata CSV used eight significant digits. Preserve that
        # precision rather than silently using a different estimation dataset.
        values = [format(float(row[field]), ".8g") for field in NUMERIC_FIELDS]
        writer.writerow((int(identifier), *values, country_codes[str(row["wbcode"])], row["cluster"]))
    content = stream.getvalue().encode("utf-8")
    output = Path(output).expanduser().resolve()
    if output.exists():
        if output.read_bytes() != content:
            raise FileExistsError("Output exists with different contents; choose a fresh data path")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE.parent / "outputs/data/greek_econometrica_data.csv")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded official ZIP")
    args = parser.parse_args(argv)
    if args.archive is not None:
        archive = args.archive.expanduser().read_bytes()
    else:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "bbp-replication/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = response.read()
    result = prepare(archive, args.output)
    print(f"Prepared original nighttime-lights data: {result}")
    print("Source: Michalopoulos and Papaioannou (2013), Econometrica, DOI:10.3982/ECTA9613")
    print("The original data remain subject to the source's terms; the repository license covers its code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
