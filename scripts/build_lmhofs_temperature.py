#!/usr/bin/env python3
"""Build a mobile-friendly west-shore Lake Michigan surface-temperature overlay.

Primary source: NOAA LMHOFS FVCOM nowcast fields.
Output:
  lmhofs-temperature.png   transparent georeferenced temperature surface
  lmhofs-temperature.json  metadata consumed by PierBite-Conditions-Map.html

The script prefers NOAA CO-OPS OPeNDAP so only the variables we need are read.
If OPeNDAP is unavailable, it falls back to downloading the matching NOMADS
NetCDF file and reads it locally.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import requests

OUT_PNG = Path("lmhofs-temperature.png")
OUT_JSON = Path("lmhofs-temperature.json")

BBOX = {
    "west": -88.10,
    "east": -86.35,
    "south": 43.20,
    "north": 45.75,
}

COLOR_STOPS = [
    (78 / 255, 42 / 255, 102 / 255),
    (20 / 255, 153 / 255, 166 / 255),
    (125 / 255, 221 / 255, 9 / 255),
    (1.0, 254 / 255, 12 / 255),
    (249 / 255, 164 / 255, 31 / 255),
    (247 / 255, 11 / 255, 66 / 255),
    (154 / 255, 5 / 255, 5 / 255),
]

CMAP = LinearSegmentedColormap.from_list(
    "pierbite_temperature",
    COLOR_STOPS,
    N=256
)

THREDDS_ROOTS = [
    "https://opendap.co-ops.nos.noaa.gov/threddsdev/dodsC/NOAA/LMHOFS/MODELS",
    "https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LMHOFS/MODELS",
]

NOMADS_ROOT = (
    "https://nomads.ncep.noaa.gov/"
    "pub/data/nccf/com/nosofs/prod"
)


@dataclass
class ModelField:
    lon: np.ndarray
    lat: np.ndarray
    temp_c: np.ndarray
    triangles: np.ndarray
    wet_nodes: np.ndarray | None
    valid_time: datetime | None
    source_url: str
    cycle_utc: str
    filename: str
    surface_layer_index: int


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def candidate_runs(
    now: datetime
) -> Iterable[tuple[datetime, int, int]]:

    for day_back in range(0, 3):

        day = (
            now -
            timedelta(days=day_back)
        ).date()

        for cycle in (18, 12, 6, 0):

            run_dt = datetime(
                day.year,
                day.month,
                day.day,
                cycle,
                tzinfo=timezone.utc
            )

            if (
                run_dt >
                now -
                timedelta(minutes=90)
            ):
                continue

            for step in (6, 5, 4, 3, 2, 1, 0):
                yield run_dt, cycle, step


def field_filename(
    run_dt: datetime,
    cycle: int,
    step: int
) -> str:

    d = run_dt.strftime("%Y%m%d")

    return (
        f"lmhofs.t{cycle:02d}z."
        f"{d}.fields.n{step:03d}.nc"
    )


def thredds_url(
    run_dt: datetime,
    filename: str,
    root: str
) -> str:

    return (
        f"{root}/"
        f"{run_dt:%Y/%m/%d}/"
        f"{filename}"
    )


def nomads_url(
    run_dt: datetime,
    filename: str
) -> str:

    return (
        f"{NOMADS_ROOT}/"
        f"lmhofs.{run_dt:%Y%m%d}/"
        f"{filename}"
    )


def remote_exists(
    url: str,
    timeout: int = 12
) -> bool:

    try:

        r = requests.get(
            url + ".dds",
            timeout=timeout
        )

        return (
            r.status_code == 200
            and
            "Dataset" in r.text
        )

    except requests.RequestException:
        return False


def _as_array(var) -> np.ndarray:

    arr = np.asanyarray(
        var[:]
    )

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    return np.asarray(arr)


def _surface_index(ds) -> int:

    sig = ds.variables.get(
        "siglay"
    )

    if sig is None:
        return 0

    vals = np.asanyarray(
        sig[:, 0]
    ).astype(float)

    if np.ma.isMaskedArray(vals):
        vals = vals.filled(np.nan)

    finite = np.isfinite(vals)

    if not np.any(finite):
        return 0

    idxs = np.where(
        finite
    )[0]

    return int(
        idxs[
            np.argmin(
                np.abs(
                    vals[finite]
                )
            )
        ]
    )


def _valid_time(ds) -> datetime | None:

    tvar = ds.variables.get(
        "time"
    )

    if (
        tvar is None
        or
        tvar.size == 0
    ):
        return None

    try:

        value = float(
            np.asanyarray(
                tvar[:]
            ).ravel()[0]
        )

        cal = getattr(
            tvar,
            "calendar",
            "standard"
        )

        import netCDF4

        d = netCDF4.num2date(
            value,
            tvar.units,
            calendar=cal,
            only_use_cftime_datetimes=False
        )

        if isinstance(
            d,
            datetime
        ):

            if d.tzinfo is None:

                d = d.replace(
                    tzinfo=timezone.utc
                )

            return d.astimezone(
                timezone.utc
            )

    except Exception:
        pass

    return None


def read_dataset(
    ds,
    source_url: str,
    cycle_utc: str,
    filename: str
) -> ModelField:

    required = (
        "lon",
        "lat",
        "temp",
        "nv"
    )

    missing = [
        v
        for v in required
        if v not in ds.variables
    ]

    if missing:

        raise RuntimeError(
            "LMHOFS file missing variables: "
            +
            ", ".join(missing)
        )

    surface_idx = _surface_index(
        ds
    )

    lon = _as_array(
        ds.variables["lon"]
    ).astype(float).ravel()

    # NOAA longitude variables are in degrees_east. Depending on the
    # dataset/service, west longitudes may be represented either as
    # negative degrees or as 0..360 degrees. PierBite's map bounds use
    # -180..180, so normalize safely to that convention.
    lon = (
        (lon + 180.0)
        % 360.0
    ) - 180.0

    lat = _as_array(
        ds.variables["lat"]
    ).astype(float).ravel()

    tvar = ds.variables[
        "temp"
    ]

    temp_c = np.asanyarray(
        tvar[
            0,
            surface_idx,
            :
        ]
    ).astype(float).ravel()

    if np.ma.isMaskedArray(
        temp_c
    ):
        temp_c = temp_c.filled(
            np.nan
        )

    nv = _as_array(
        ds.variables["nv"]
    ).astype(np.int64)

    if nv.ndim != 2:

        raise RuntimeError(
            f"Unexpected nv shape: "
            f"{nv.shape}"
        )

    if nv.shape[0] == 3:

        triangles = nv.T.copy()

    elif nv.shape[1] == 3:

        triangles = nv.copy()

    else:

        raise RuntimeError(
            f"Unexpected nv shape: "
            f"{nv.shape}"
        )

    if triangles.min() >= 1:
        triangles -= 1

    wet = None

    if "wet_nodes" in ds.variables:

        w = np.asanyarray(
            ds.variables[
                "wet_nodes"
            ][0, :]
        ).ravel()

        if np.ma.isMaskedArray(
            w
        ):
            w = w.filled(0)

        wet = np.asarray(
            w
        ).astype(bool)

    return ModelField(
        lon=lon,
        lat=lat,
        temp_c=temp_c,
        triangles=triangles,
        wet_nodes=wet,
        valid_time=_valid_time(ds),
        source_url=source_url,
        cycle_utc=cycle_utc,
        filename=filename,
        surface_layer_index=surface_idx,
    )


def load_from_opendap(
    now: datetime
) -> ModelField | None:

    errors: list[str] = []

    for (
        run_dt,
        cycle,
        step
    ) in candidate_runs(now):

        filename = field_filename(
            run_dt,
            cycle,
            step
        )

        cycle_utc = (
            f"{run_dt:%Y-%m-%d} "
            f"{cycle:02d}Z"
        )

        for root in THREDDS_ROOTS:

            url = thredds_url(
                run_dt,
                filename,
                root
            )

            if not remote_exists(
                url
            ):
                continue

            try:

                print(
                    "Trying NOAA OPeNDAP: "
                    + url
                )

                import netCDF4

                with netCDF4.Dataset(
                    url
                ) as ds:

                    field = read_dataset(
                        ds,
                        url,
                        cycle_utc,
                        filename
                    )

                print(
                    "Using NOAA OPeNDAP field: "
                    + filename
                )

                return field

            except Exception as exc:

                errors.append(
                    f"{url}: {exc}"
                )

                print(
                    "OPeNDAP read failed: "
                    + str(exc),
                    file=sys.stderr
                )

    if errors:

        print(
            "OPeNDAP attempts failed; "
            "will try NOMADS download.",
            file=sys.stderr
        )

    return None


def head_ok(
    url: str,
    timeout: int = 15
) -> bool:

    try:

        r = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True
        )

        return (
            r.status_code == 200
        )

    except requests.RequestException:
        return False


def download_file(
    url: str,
    destination: Path
) -> None:

    with requests.get(
        url,
        stream=True,
        timeout=(20, 180)
    ) as r:

        r.raise_for_status()

        with destination.open(
            "wb"
        ) as f:

            for chunk in r.iter_content(
                chunk_size=
                1024 * 1024
            ):

                if chunk:
                    f.write(chunk)


def load_from_nomads(
    now: datetime
) -> ModelField:

    for (
        run_dt,
        cycle,
        step
    ) in candidate_runs(now):

        filename = field_filename(
            run_dt,
            cycle,
            step
        )

        url = nomads_url(
            run_dt,
            filename
        )

        if not head_ok(
            url
        ):
            continue

        cycle_utc = (
            f"{run_dt:%Y-%m-%d} "
            f"{cycle:02d}Z"
        )

        with tempfile.TemporaryDirectory(
            prefix=
            "pierbite-lmhofs-"
        ) as td:

            local = (
                Path(td)
                /
                filename
            )

            print(
                "Downloading NOAA "
                "NOMADS fallback: "
                + url
            )

            download_file(
                url,
                local
            )

            import netCDF4

            with netCDF4.Dataset(
                local
            ) as ds:

                return read_dataset(
                    ds,
                    url,
                    cycle_utc,
                    filename
                )

    raise RuntimeError(
        "Could not find a recent "
        "LMHOFS nowcast through "
        "OPeNDAP or NOMADS."
    )


def load_latest_field(
    now: datetime
) -> ModelField:

    field = load_from_opendap(
        now
    )

    if field is not None:
        return field

    return load_from_nomads(
        now
    )


def west_shore_focus(
    lon: np.ndarray,
    lat: np.ndarray
) -> np.ndarray:

    return ~(
        (lat > 44.55)
        &
        (lon < -87.62)
    )


def fahrenheit(
    c: np.ndarray
) -> np.ndarray:

    return (
        c *
        9.0 /
        5.0 +
        32.0
    )


def build_mask(
    field: ModelField
) -> tuple[
    np.ndarray,
    np.ndarray
]:

    lon = field.lon
    lat = field.lat

    temp_f = fahrenheit(
        field.temp_c
    )

    node_good = (
        np.isfinite(lon)
        &
        np.isfinite(lat)
        &
        np.isfinite(temp_f)
        &
        (temp_f > 30.0)
        &
        (temp_f < 100.0)
        &
        west_shore_focus(
            lon,
            lat
        )
    )

    if (
        field.wet_nodes
        is not None
        and
        field.wet_nodes.size
        ==
        node_good.size
    ):

        node_good &= field.wet_nodes

    tri = field.triangles

    valid_index = (
        (tri >= 0)
        &
        (tri < len(lon))
    )

    triangle_good = np.all(
        valid_index,
        axis=1
    )

    safe_tri = tri.copy()

    safe_tri[
        ~valid_index
    ] = 0

    tri_node_good = np.all(
        node_good[
            safe_tri
        ],
        axis=1
    )

    cx = np.mean(
        lon[safe_tri],
        axis=1
    )

    cy = np.mean(
        lat[safe_tri],
        axis=1
    )

    inside = (
        (
            cx >=
            BBOX["west"] -
            0.05
        )
        &
        (
            cx <=
            BBOX["east"] +
            0.05
        )
        &
        (
            cy >=
            BBOX["south"] -
            0.05
        )
        &
        (
            cy <=
            BBOX["north"] +
            0.05
        )
    )

    lons = lon[
        safe_tri
    ]

    lats = lat[
        safe_tri
    ]

    span_lon = (
        np.max(
            lons,
            axis=1
        )
        -
        np.min(
            lons,
            axis=1
        )
    )

    span_lat = (
        np.max(
            lats,
            axis=1
        )
        -
        np.min(
            lats,
            axis=1
        )
    )

    local_size = (
        (span_lon < 0.20)
        &
        (span_lat < 0.20)
    )

    mask = ~(
        triangle_good
        &
        tri_node_good
        &
        inside
        &
        local_size
    )

    return temp_f, mask


def display_scale(
    temp_f: np.ndarray,
    field: ModelField
) -> tuple[
    float,
    float,
    float,
    float
]:

    in_box = (
        np.isfinite(temp_f)
        &
        (
            field.lon >=
            BBOX["west"]
        )
        &
        (
            field.lon <=
            BBOX["east"]
        )
        &
        (
            field.lat >=
            BBOX["south"]
        )
        &
        (
            field.lat <=
            BBOX["north"]
        )
        &
        (temp_f > 30.0)
        &
        (temp_f < 100.0)
        &
        west_shore_focus(
            field.lon,
            field.lat
        )
    )

    vals = temp_f[
        in_box
    ]

    if vals.size < 100:

        finite_lon = field.lon[np.isfinite(field.lon)]
        finite_lat = field.lat[np.isfinite(field.lat)]
        finite_temp = temp_f[np.isfinite(temp_f)]

        lon_range = (
            f"{float(np.nanmin(finite_lon)):.3f}.."
            f"{float(np.nanmax(finite_lon)):.3f}"
            if finite_lon.size
            else "none"
        )

        lat_range = (
            f"{float(np.nanmin(finite_lat)):.3f}.."
            f"{float(np.nanmax(finite_lat)):.3f}"
            if finite_lat.size
            else "none"
        )

        temp_range = (
            f"{float(np.nanmin(finite_temp)):.2f}.."
            f"{float(np.nanmax(finite_temp)):.2f} F"
            if finite_temp.size
            else "none"
        )

        raise RuntimeError(
            f"Only {vals.size} valid "
            "LMHOFS surface-temperature "
            "nodes in the map box. "
            f"Dataset ranges after normalization: "
            f"lon {lon_range}, "
            f"lat {lat_range}, "
            f"temp {temp_range}."
        )

    actual_min = float(
        np.nanmin(vals)
    )

    actual_max = float(
        np.nanmax(vals)
    )

    lo = float(
        np.nanpercentile(
            vals,
            1.0
        )
    )

    hi = float(
        np.nanpercentile(
            vals,
            99.0
        )
    )

    lo = math.floor(lo)
    hi = math.ceil(hi)

    if (
        hi -
        lo <
        4.0
    ):

        mid = (
            hi +
            lo
        ) / 2.0

        lo = math.floor(
            mid -
            2.0
        )

        hi = math.ceil(
            mid +
            2.0
        )

    lo = max(
        30.0,
        lo
    )

    hi = min(
        90.0,
        hi
    )

    return (
        lo,
        hi,
        actual_min,
        actual_max
    )


def render_overlay(
    field: ModelField,
    out_png: Path,
    out_json: Path
) -> dict:

    (
        temp_f,
        mask
    ) = build_mask(
        field
    )

    (
        vmin,
        vmax,
        actual_min,
        actual_max
    ) = display_scale(
        temp_f,
        field
    )

    triang = mtri.Triangulation(
        field.lon,
        field.lat,
        triangles=
        field.triangles
    )

    triang.set_mask(
        mask
    )

    fig = plt.figure(
        figsize=(8.0, 11.66),
        dpi=150,
        frameon=False
    )

    ax = fig.add_axes(
        [0, 0, 1, 1]
    )

    ax.set_facecolor(
        (0, 0, 0, 0)
    )

    fig.patch.set_alpha(
        0
    )

    levels = np.linspace(
        vmin,
        vmax,
        41
    )

    ax.tricontourf(
        triang,
        temp_f,
        levels=levels,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        extend="both",
        antialiased=True,
    )

    ax.set_xlim(
        BBOX["west"],
        BBOX["east"]
    )

    ax.set_ylim(
        BBOX["south"],
        BBOX["north"]
    )

    ax.axis(
        "off"
    )

    fig.savefig(
        out_png,
        transparent=True,
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(
        fig
    )

    unmasked_triangles = int(
        np.count_nonzero(
            ~mask
        )
    )

    crop_nodes = int(
        np.count_nonzero(
            (
                field.lon >=
                BBOX["west"]
            )
            &
            (
                field.lon <=
                BBOX["east"]
            )
            &
            (
                field.lat >=
                BBOX["south"]
            )
            &
            (
                field.lat <=
                BBOX["north"]
            )
            &
            np.isfinite(temp_f)
        )
    )

    generated = utcnow()

    meta = {

        "product":
        "NOAA LMHOFS surface "
        "water temperature",

        "generated_at_utc":
        generated
        .isoformat()
        .replace(
            "+00:00",
            "Z"
        ),

        "valid_time_utc":
        (
            field.valid_time
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            )
            if
            field.valid_time
            else
            None
        ),

        "cycle_utc":
        field.cycle_utc,

        "source_file":
        field.filename,

        "source_url":
        field.source_url,

        "surface_sigma_layer_index":
        field.surface_layer_index,

        "bounds":
        BBOX,

        "scale_min_f":
        round(
            vmin,
            1
        ),

        "scale_max_f":
        round(
            vmax,
            1
        ),

        "actual_min_f":
        round(
            actual_min,
            2
        ),

        "actual_max_f":
        round(
            actual_max,
            2
        ),

        "crop_node_count":
        crop_nodes,

        "rendered_triangle_count":
        unmasked_triangles,

        "palette": [
            "#4e2a66",
            "#1499a6",
            "#7ddd09",
            "#fffe0c",
            "#f9a41f",
            "#f70b42",
            "#9a0505"
        ],

        "scale_note":
        "Color contrast is adjusted "
        "to the current west-shore "
        "LMHOFS temperature range."
    }

    out_json.write_text(
        json.dumps(
            meta,
            indent=2
        )
        +
        "\n",
        encoding="utf-8"
    )

    if (
        crop_nodes < 1000
        or
        unmasked_triangles < 1000
    ):

        raise RuntimeError(
            "Safety check failed: "
            f"only {crop_nodes} nodes / "
            f"{unmasked_triangles} "
            "triangles rendered."
        )

    if (
        actual_max -
        actual_min <
        0.05
    ):

        raise RuntimeError(
            "Safety check failed: "
            "LMHOFS field has essentially "
            "no spatial temperature "
            "variation."
        )

    if (
        not out_png.exists()
        or
        out_png.stat().st_size
        <
        50_000
    ):

        raise RuntimeError(
            "Safety check failed: "
            "generated temperature PNG "
            "is unexpectedly small."
        )

    return meta


def self_test() -> None:

    nx = 70
    ny = 90

    xs = np.linspace(
        BBOX["west"],
        BBOX["east"],
        nx
    )

    ys = np.linspace(
        BBOX["south"],
        BBOX["north"],
        ny
    )

    xx, yy = np.meshgrid(
        xs,
        ys
    )

    lon = xx.ravel()
    lat = yy.ravel()

    cold = (
        10.0
        *
        np.exp(
            -(
                (
                    (
                        xx +
                        87.55
                    )
                    /
                    0.18
                )
                ** 2
                +
                (
                    (
                        yy -
                        44.25
                    )
                    /
                    0.45
                )
                ** 2
            )
        )
    )

    celsius = (
        20.5
        +
        1.8
        *
        (
            xx -
            BBOX["west"]
        )
        -
        cold
    )

    temp_c = celsius.ravel()

    tris = []

    for j in range(
        ny - 1
    ):

        for i in range(
            nx - 1
        ):

            a = (
                j *
                nx +
                i
            )

            b = a + 1
            c = a + nx
            d = c + 1

            tris.append(
                (a, b, d)
            )

            tris.append(
                (a, d, c)
            )

    field = ModelField(
        lon=lon,
        lat=lat,
        temp_c=temp_c,
        triangles=
        np.asarray(
            tris,
            dtype=np.int64
        ),
        wet_nodes=
        np.ones_like(
            lon,
            dtype=bool
        ),
        valid_time=
        utcnow(),
        source_url=
        "self-test",
        cycle_utc=
        "self-test",
        filename=
        "self-test.nc",
        surface_layer_index=0
    )

    with tempfile.TemporaryDirectory(
        prefix=
        "pierbite-selftest-"
    ) as td:

        p = (
            Path(td)
            /
            "test.png"
        )

        j = (
            Path(td)
            /
            "test.json"
        )

        meta = render_overlay(
            field,
            p,
            j
        )

        if (
            meta["actual_max_f"]
            -
            meta["actual_min_f"]
            <
            5
        ):

            raise RuntimeError(
                "Self-test did not "
                "preserve the synthetic "
                "thermal contrast."
            )

        print(
            "LMHOFS renderer "
            "self-test passed."
        )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=
        "run renderer validation "
        "without network access"
    )

    args = parser.parse_args()

    if args.self_test:

        self_test()

        return 0

    now = utcnow()

    print(
        "PierBite LMHOFS build started: "
        +
        now.isoformat()
    )

    field = load_latest_field(
        now
    )

    meta = render_overlay(
        field,
        OUT_PNG,
        OUT_JSON
    )

    print(
        json.dumps(
            {
                "source_file":
                meta["source_file"],

                "valid_time_utc":
                meta[
                    "valid_time_utc"
                ],

                "scale_f": [
                    meta["scale_min_f"],
                    meta["scale_max_f"]
                ],

                "actual_f": [
                    meta["actual_min_f"],
                    meta["actual_max_f"]
                ],

                "nodes":
                meta[
                    "crop_node_count"
                ],

                "triangles":
                meta[
                    "rendered_triangle_count"
                ],

                "png_bytes":
                OUT_PNG.stat().st_size
            },
            indent=2
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
