import json
import math
import re
import tomllib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

UJI_ZIP_URL = "https://archive.ics.uci.edu/static/public/343/ujiindoorloc%2Bmag.zip"
MARKER_RE = re.compile(r"<\d+>")

# ============================================================
# Private Definitions
# ============================================================

# --- Private constants (used by get_map(source="uji")) ---


# --- Private helpers used by get_map(source="uji"): dataset bootstrap ---
def _download_uji_zip(zip_path, force_download=False):
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and not force_download:
        return zip_path

    # TODO: Add checksum validation for dataset integrity.
    urlretrieve(UJI_ZIP_URL, zip_path)
    return zip_path


def _extract_uji_zip(zip_path, extract_dir, force_extract=False):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    marker = extract_dir / ".extracted_ok"
    if marker.exists() and not force_extract:
        return extract_dir

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    marker.write_text("ok", encoding="utf-8")
    return extract_dir


# --- Private helpers used by get_map(source="uji"): configuration ---
def _load_map_builder_cfg(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    tool_cfg = config.get("tool", {})
    if isinstance(tool_cfg, dict):
        map_cfg = tool_cfg.get("map_builder", {})
        if isinstance(map_cfg, dict):
            return map_cfg
    return {}


# --- Private helpers used by get_map(source="uji"): UJI text parsing ---
def _is_sensor_row(line):
    parts = line.strip().split()
    if len(parts) < 10:
        return False
    try:
        int(float(parts[0]))
        [float(v) for v in parts[1:10]]
        return True
    except ValueError:
        return False


def _is_segment_row(line):
    parts = line.strip().split()
    if len(parts) != 6:
        return False
    try:
        [float(v) for v in parts]
        return True
    except ValueError:
        return False


def _parse_uji_file(path):
    sensor_rows = []
    segment_rows = []

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if MARKER_RE.search(line):
                continue
            if _is_sensor_row(line):
                parts = line.split()
                mx, my, mz = float(parts[1]), float(parts[2]), float(parts[3])
                sensor_rows.append((mx, my, mz))
                continue
            if _is_segment_row(line):
                p = [float(v) for v in line.split()]
                lat1, lon1, lat2, lon2 = p[0], p[1], p[2], p[3]
                i0, i1 = int(round(p[4])), int(round(p[5]))
                segment_rows.append((lat1, lon1, lat2, lon2, i0, i1))

    if not sensor_rows or not segment_rows:
        return np.array([]), np.array([]), np.array([])

    n = len(sensor_rows)
    lats = np.full(n, np.nan, dtype=float)
    lons = np.full(n, np.nan, dtype=float)
    mags = np.empty(n, dtype=float)

    for i, (mx, my, mz) in enumerate(sensor_rows):
        mags[i] = math.sqrt(mx * mx + my * my + mz * mz)

    for lat1, lon1, lat2, lon2, i0, i1 in segment_rows:
        if i1 < i0:
            i0, i1 = i1, i0
            lat1, lon1, lat2, lon2 = lat2, lon2, lat1, lon1
        i0 = max(i0, 0)
        i1 = min(i1, n - 1)
        if i0 > i1:
            continue
        count = i1 - i0 + 1
        lats[i0 : i1 + 1] = np.linspace(lat1, lat2, count)
        lons[i0 : i1 + 1] = np.linspace(lon1, lon2, count)

    valid = np.isfinite(lats) & np.isfinite(lons)
    return lats[valid], lons[valid], mags[valid]


# --- Private helpers used by get_map(source="uji"): geometry and sampling ---
def _collect_uji_points(dataset_root):
    dataset_root = Path(dataset_root)
    paths = list((dataset_root / "lines").rglob("*.txt"))
    paths += list((dataset_root / "curves").rglob("*.txt"))

    lat_all = []
    lon_all = []
    mag_all = []

    for path in sorted(paths):
        lat, lon, mag = _parse_uji_file(path)
        if lat.size == 0:
            continue
        lat_all.append(lat)
        lon_all.append(lon)
        mag_all.append(mag)

    if not lat_all:
        raise RuntimeError("No valid UJI training points parsed from lines/curves.")

    return np.concatenate(lat_all), np.concatenate(lon_all), np.concatenate(mag_all)


def _latlon_to_xy(lat, lon, lat0, lon0):
    radius = 6378137.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = radius * dlon * np.cos(np.radians((lat + lat0) * 0.5))
    y = radius * dlat
    return x, y


def _reduce_points_for_kriging(x, y, z, max_points, seed):
    if x.size <= max_points:
        return x, y, z
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_points, replace=False)
    return x[idx], y[idx], z[idx]


# --- Private helpers shared by get_map(source="uji") and visualize(...) ---
def _fit_ordinary_kriging(x, y, z, variogram_model):
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise ImportError(
            "pykrige is required for Kriging interpolation. Install it with: pip install pykrige"
        ) from exc

    return OrdinaryKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )


def _predict_continuous_grid(ok_model, min_x, max_x, min_y, max_y, resolution):
    grid_x = np.arange(min_x, max_x + resolution, resolution, dtype=float)
    grid_y = np.arange(min_y, max_y + resolution, resolution, dtype=float)
    grid_z, grid_ss = ok_model.execute("grid", grid_x, grid_y)
    return grid_x, grid_y, np.asarray(grid_z, dtype=float), np.asarray(grid_ss, dtype=float)


# --- Private helper used by get_map(source="uji"): preview artifact plotting ---
def _plot_grid_map(grid_x, grid_y, grid_z, output_png):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        ) from exc

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    extent = [float(grid_x[0]), float(grid_x[-1]), float(grid_y[0]), float(grid_y[-1])]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    im = ax.imshow(
        np.asarray(grid_z, dtype=float),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title("UJI Magnetic Magnitude Map (Continuous Kriging Preview)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnetic Magnitude")
    fig.tight_layout()
    # Overwrite if existing.
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


# --- Private builder used by get_map(source="uji"): continuous UJI map ---
def _build_uji_continuous_map(
    extracted_root,
    output_model_npz,
    output_preview_npz,
    output_json,
    output_png,
    preview_resolution,
    max_kriging_points,
    variogram_model,
    seed,
):
    lat, lon, mag = _collect_uji_points(extracted_root)
    lat0 = float(lat.min())
    lon0 = float(lon.min())
    x, y = _latlon_to_xy(lat, lon, lat0=lat0, lon0=lon0)
    x_train, y_train, z_train = _reduce_points_for_kriging(
        x, y, mag, max_points=max_kriging_points, seed=seed
    )

    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_y, max_y = float(np.min(y)), float(np.max(y))
    ok = _fit_ordinary_kriging(
        x=x_train,
        y=y_train,
        z=z_train,
        variogram_model=variogram_model,
    )
    grid_x, grid_y, grid_z, grid_ss = _predict_continuous_grid(
        ok_model=ok,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        resolution=preview_resolution,
    )

    output_model_npz = Path(output_model_npz)
    output_preview_npz = Path(output_preview_npz)
    output_json = Path(output_json)
    output_png = Path(output_png)
    output_model_npz.parent.mkdir(parents=True, exist_ok=True)
    output_preview_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_model_npz,
        mode=np.array(["continuous_ordinary_kriging"]),
        x_train=np.asarray(x_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        z_train=np.asarray(z_train, dtype=float),
        variogram_model=np.array([variogram_model]),
        min_x=np.array([min_x], dtype=float),
        max_x=np.array([max_x], dtype=float),
        min_y=np.array([min_y], dtype=float),
        max_y=np.array([max_y], dtype=float),
        origin_lat=np.array([lat0], dtype=float),
        origin_lon=np.array([lon0], dtype=float),
    )
    np.savez_compressed(
        output_preview_npz,
        grid_magnitude=np.asarray(grid_z, dtype=float),
        grid_variance=np.asarray(grid_ss, dtype=float),
        grid_x=grid_x,
        grid_y=grid_y,
        resolution=np.array([preview_resolution], dtype=float),
    )
    _plot_grid_map(grid_x=grid_x, grid_y=grid_y, grid_z=grid_z, output_png=output_png)

    metadata = {
        "source": "uji",
        "continuous_map": True,
        "dataset_root": str(extracted_root),
        "points_total": int(x.size),
        "points_used_for_kriging": int(x_train.size),
        "variogram_model": variogram_model,
        "preview_resolution": float(preview_resolution),
        "bounds_xy_m": [min_x, max_x, min_y, max_y],
        "preview_grid_shape": [int(len(grid_y)), int(len(grid_x))],
        "origin_latlon": [lat0, lon0],
        "output_model_npz": str(output_model_npz),
        "output_preview_npz": str(output_preview_npz),
        "output_png": str(output_png),
    }
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["output_json"] = str(output_json)
    return metadata


# --- Private builder used by get_map(source="own"): interface contract ---
def _build_own_map_interface(
    own_grid_array=None,
    own_grid_map_path=None,
    own_grid_format="array",
    own_grid_meta=None,
):
    matrix_array = None
    matrix_shape = None
    matrix_valid = False
    if own_grid_array is not None:
        matrix_array = np.asarray(own_grid_array, dtype=float)
        matrix_valid = matrix_array.ndim == 2 and matrix_array.size > 0
        if matrix_valid:
            matrix_shape = [int(matrix_array.shape[0]), int(matrix_array.shape[1])]

    grid_path = Path(own_grid_map_path) if own_grid_map_path is not None else None
    default_meta = {
        "cell_size_m": None,
        "origin_xy_m": [0.0, 0.0],
        "x_axis_direction": "east",
        "y_axis_direction": "north",
        "mag_unit": "uT",
    }
    merged_meta = dict(default_meta)
    if isinstance(own_grid_meta, dict):
        merged_meta.update(own_grid_meta)

    grid_input_specs = {
        "array": {
            "description": "Direct in-memory 2D matrix (list[list[float]] or np.ndarray)",
            "shape": "(H, W)",
            "dtype": "float",
            "semantics": "matrix[row, col] -> magnetic magnitude at grid cell",
        },
        "npy_matrix": {
            "description": "2D magnetic map matrix as plain numpy array",
            "shape": "(H, W)",
            "dtype": "float",
            "semantics": "matrix[row, col] -> magnetic magnitude at grid cell",
        },
        "npz_matrix": {
            "description": "2D magnetic map matrix packed in npz",
            "required_keys": ["grid_magnitude"],
            "optional_keys": ["mask", "grid_variance"],
            "shape": "(H, W)",
        },
        "csv_matrix": {
            "description": "2D matrix stored as CSV table",
            "shape": "(H, W)",
            "note": "No header recommended; each row is one grid row.",
        },
    }

    return {
        "source": "own",
        "status": "interface_only",
        "grid_array": matrix_array.tolist() if matrix_valid else None,
        "array_input": {
            "provided": own_grid_array is not None,
            "valid_2d": matrix_valid,
            "shape": matrix_shape,
        },
        "grid_map_contract": {
            "selected_format": own_grid_format,
            "path": str(grid_path) if grid_path is not None else None,
            "exists": grid_path.exists() if grid_path is not None else None,
            "supported_formats": grid_input_specs,
            "meta": merged_meta,
            "matrix_convention": {
                "indexing": "row-major",
                "value": "magnetic magnitude",
                "world_mapping": "x = origin_x + col * cell_size; y = origin_y + row * cell_size",
            },
        },
        "next_step": "Use own_grid_array for direct editable matrix input; keep file path only as optional fallback.",
    }


# ============================================================
# Public Definitions
# ============================================================

# --- Public API: map factory (called by Initializer.create_context / temp scripts) ---
def get_map(
    source="uji",
    data_root="data/raw",
    config_path="pyproject.toml",
    force_download=False,
    force_extract=False,
    own_grid_array=None,
    own_grid_map_path=None,
    own_grid_format="array",
    own_grid_meta=None,
):
    source = source.lower()

    if source == "uji":
        cfg = _load_map_builder_cfg(config_path)
        preview_resolution = float(cfg.get("preview_resolution", 1.0))
        max_kriging_points = int(cfg.get("max_kriging_points", 2500))
        seed = int(cfg.get("seed", 42))
        variogram_model = str(cfg.get("variogram_model", "spherical"))
        output_model_npz = cfg.get("output_model_npz", "data/processed/uji_mag_model_kriging.npz")
        output_preview_npz = cfg.get(
            "output_preview_npz",
            "data/processed/uji_mag_grid_preview_kriging.npz",
        )
        output_json = cfg.get("output_json", "data/processed/uji_mag_grid_kriging_meta.json")
        output_png = cfg.get("output_png", "data/processed/uji_mag_grid_kriging.png")

        data_root_path = Path(data_root)
        uji_root = data_root_path / "uji_indoorloc_mag"
        zip_path = uji_root / "ujiindoorloc+mag.zip"
        extract_dir = uji_root / "extracted"
        _download_uji_zip(zip_path, force_download=force_download)
        _extract_uji_zip(zip_path, extract_dir, force_extract=force_extract)
        extracted_root = extract_dir / "UJIIndoorLoc-Mag" / "UJIIndoorLoc-Mag"

        map_info = _build_uji_continuous_map(
            extracted_root=extracted_root,
            output_model_npz=output_model_npz,
            output_preview_npz=output_preview_npz,
            output_json=output_json,
            output_png=output_png,
            preview_resolution=preview_resolution,
            max_kriging_points=max_kriging_points,
            variogram_model=variogram_model,
            seed=seed,
        )
        map_info["zip_path"] = str(zip_path)
        map_info["extract_dir"] = str(extract_dir)
        return map_info

    if source == "own":
        return _build_own_map_interface(
            own_grid_array=own_grid_array,
            own_grid_map_path=own_grid_map_path,
            own_grid_format=own_grid_format,
            own_grid_meta=own_grid_meta,
        )

    raise ValueError(f"Unsupported map source: {source}")


# --- Public API placeholders used by Experiment.run() ---
# TODO: Provide the ground-truth route for a test run.
def get_true_route():
    pass


# TODO: Return test length (number of sensor frames to consume).
def get_test_len():
    pass


# TODO: Fetch one frame of sensor data: magnetometer, accelerometer, gyroscope.
def get_sensor():
    pass


# TODO: Determine whether buffered sensor samples contain a completed step.
def judge_step(samples):
    pass


# TODO: Estimate step length from buffered samples.
def get_step_len(samples):
    pass


# TODO: Estimate heading angle from buffered samples.
def get_heading_angle(samples):
    pass


# TODO: Extract geomagnetic feature/value used by the filter.
def get_mag():
    pass


# TODO: Run one particle-filter update step and return updated position.
def PF(step_len, heading_angle, geomag_list, pf_state):
    pass


# --- Public API: visualization router (called by Experiment.run / temp scripts) ---
# TODO: Visualize estimated track vs. route on the map.
def visualize(pos_list=None, route=None, geomag_map=None, mode="track", vis_resolution=0.2):
    if mode == "ujimap":
        if not isinstance(geomag_map, dict) or "output_model_npz" not in geomag_map:
            raise ValueError("`geomag_map` must contain `output_model_npz` for mode='ujimap'.")

        model = np.load(Path(geomag_map["output_model_npz"]))
        x_train = np.asarray(model["x_train"], dtype=float)
        y_train = np.asarray(model["y_train"], dtype=float)
        z_train = np.asarray(model["z_train"], dtype=float)
        variogram_model = str(model["variogram_model"][0])
        min_x = float(model["min_x"][0])
        max_x = float(model["max_x"][0])
        min_y = float(model["min_y"][0])
        max_y = float(model["max_y"][0])

        grid_x = np.arange(min_x, max_x + vis_resolution, vis_resolution, dtype=float)
        grid_y = np.arange(min_y, max_y + vis_resolution, vis_resolution, dtype=float)
        ok = _fit_ordinary_kriging(
            x=x_train,
            y=y_train,
            z=z_train,
            variogram_model=variogram_model,
        )
        grid_z, _ = ok.execute("grid", grid_x, grid_y)
        grid_z = np.asarray(grid_z, dtype=float)

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualization. Install it with: pip install matplotlib"
            ) from exc

        xx, yy = np.meshgrid(grid_x, grid_y)
        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        contour = ax.contourf(xx, yy, grid_z, levels=80, cmap="viridis")
        ax.set_title("UJI Continuous Magnetic Map (Kriging)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Magnetic Magnitude")
        fig.tight_layout()
        plt.show()
        return

    if mode == "usermap":
        if not isinstance(geomag_map, dict):
            raise ValueError("`geomag_map` must be a dict for mode='usermap'.")
        if geomag_map.get("source") != "own":
            raise ValueError("mode='usermap' requires own-map input from get_map(source='own').")
        grid_array = geomag_map.get("grid_array")
        if grid_array is None:
            raise ValueError("`grid_array` is missing. Provide a valid own_grid_array to get_map().")

        z = np.asarray(grid_array, dtype=float)
        if z.ndim != 2 or z.size == 0:
            raise ValueError("`grid_array` must be a non-empty 2D matrix.")

        meta = geomag_map.get("grid_map_contract", {}).get("meta", {})
        cell_size = float(meta.get("cell_size_m", 1.0) or 1.0)
        origin = meta.get("origin_xy_m", [0.0, 0.0])
        origin_x = float(origin[0]) if len(origin) > 0 else 0.0
        origin_y = float(origin[1]) if len(origin) > 1 else 0.0
        variogram_model = str(meta.get("variogram_model", "spherical"))

        rows, cols = z.shape
        col_coords = origin_x + np.arange(cols, dtype=float) * cell_size
        row_coords = origin_y + np.arange(rows, dtype=float) * cell_size
        xx_train, yy_train = np.meshgrid(col_coords, row_coords)
        valid = np.isfinite(z)
        x_train = xx_train[valid]
        y_train = yy_train[valid]
        z_train = z[valid]
        if x_train.size < 4:
            raise ValueError("Need at least 4 valid matrix cells for Kriging-based continuous usermap.")

        min_x = float(col_coords[0])
        max_x = float(col_coords[-1])
        min_y = float(row_coords[0])
        max_y = float(row_coords[-1])
        grid_x = np.arange(min_x, max_x + vis_resolution, vis_resolution, dtype=float)
        grid_y = np.arange(min_y, max_y + vis_resolution, vis_resolution, dtype=float)
        ok = _fit_ordinary_kriging(
            x=x_train,
            y=y_train,
            z=z_train,
            variogram_model=variogram_model,
        )
        grid_z, _ = ok.execute("grid", grid_x, grid_y)
        grid_z = np.asarray(grid_z, dtype=float)

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualization. Install it with: pip install matplotlib"
            ) from exc

        xx, yy = np.meshgrid(grid_x, grid_y)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        contour = ax.contourf(xx, yy, grid_z, levels=80, cmap="viridis")
        ax.set_title("User Continuous Magnetic Map (Kriging)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Magnetic Magnitude")
        fig.tight_layout()
        plt.show()
        return

    # TODO: Add track/route visualization for localization results.
    return
