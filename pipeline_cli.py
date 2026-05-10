# pipeline_cli.py
"""
FULL PIPELINE:
1. Filter uploaded images  → keep only histopathology images
2. Restain the filtered images
3. Classify + build train/test folder structure (predict_data_class.py)
4. Launch the Federated Learning client (proxclient.py)

RUN:
    python pipeline_cli.py --input_dir input_images --output_dir output

BEFORE RUNNING:
    Make sure the FL server is already running on localhost:8081.
    Make sure FL_SHARED_KEY is set in your shell:
        Windows PowerShell:  $env:FL_SHARED_KEY = "<hex_key>"
        Linux / macOS:       export FL_SHARED_KEY="<hex_key>"
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb

from histo_filter import preprocess_before_restaining


# ==========================================================
# DESTAIN → NPZ SAVE
# ==========================================================
def destain_and_save_npz(inp_path: Path, tmp_folder: Path):
    img = img_as_float(io.imread(str(inp_path)))
    if img.ndim == 2:
        raise ValueError(f"Input image {inp_path} is grayscale. Needs RGB.")
    if img.shape[2] == 4:
        img = img[..., :3]
    hed = rgb2hed(img)
    tmp_folder.mkdir(parents=True, exist_ok=True)
    npz_path = tmp_folder / f"{inp_path.stem}.npz"
    np.savez_compressed(str(npz_path), h=hed[..., 0], e=hed[..., 1], d=hed[..., 2])
    return npz_path


# ==========================================================
# RESTAIN
# ==========================================================
def restain_from_npz(npz_path: Path, out_path: Path, h_scale=1.1, e_scale=1.1):
    data = np.load(str(npz_path), allow_pickle=True)
    h, e, d = data["h"], data["e"], data["d"]
    hed = np.stack([h * h_scale, e * e_scale, d], axis=-1)
    rgb = np.clip(hed2rgb(hed), 0, 1)
    io.imsave(str(out_path), img_as_ubyte(rgb))
    return out_path


# ==========================================================
# PROCESS A SINGLE IMAGE
# ==========================================================
def process_single_image(img_path: Path, out_dir: Path, tmp_dir: Path):
    out_path = out_dir / f"{img_path.stem}_restained.png"
    npz = destain_and_save_npz(img_path, tmp_dir)
    result = restain_from_npz(npz, out_path)
    print(f"✔ Restained Saved: {result}", flush=True)
    return result


# ==========================================================
# SHARED SUBPROCESS RUNNER
# stdout/stderr wired directly to this terminal — nothing lost.
# ==========================================================
def run_subprocess(cmd: list, env: dict = None) -> int:
    proc = subprocess.Popen(
        cmd,
        env=env or os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    proc.wait()
    return proc.returncode


# ==========================================================
# STEP 3 — run predict_data_class.py
# ==========================================================
def run_predict_script(output_dir: Path):
    cmd = [sys.executable, "-u", "predict_data_class.py",
           "--input_dir", str(output_dir)]
    print(f"\n▶ Running: {' '.join(cmd)}\n", flush=True)
    rc = run_subprocess(cmd)
    if rc != 0:
        print(f"❌ ERROR: predict_data_class.py exited with code {rc}", flush=True)
        sys.exit(1)


# ==========================================================
# VERIFY FOLDER STRUCTURE before handing off to FL client
# Expects: output/train/<class>/ and output/test/<class>/
# ==========================================================
def verify_folder_structure(output_dir: Path) -> bool:
    print("\n🔍 Verifying folder structure for FL client...", flush=True)
    ok = True
    for split in ("train", "test"):
        split_path = output_dir / split
        if not split_path.exists():
            print(f"   ❌ Missing folder: {split_path}", flush=True)
            ok = False
            continue
        class_folders = [p for p in split_path.iterdir() if p.is_dir()]
        if not class_folders:
            print(f"   ❌ No class subfolders in: {split_path}", flush=True)
            ok = False
            continue
        total = 0
        for cls_folder in sorted(class_folders):
            imgs = [f for f in cls_folder.iterdir()
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            total += len(imgs)
            print(f"   ✔ {split:5s}/{cls_folder.name:25s}: {len(imgs):3d} images",
                  flush=True)
        if total == 0:
            print(f"   ❌ No images found under {split_path}", flush=True)
            ok = False
    return ok


# ==========================================================
# PRE-FLIGHT CHECK — validate everything before launching client
# ==========================================================
def preflight_check(output_dir: Path, client_name: str) -> bool:
    """
    Validate all prerequisites for the FL client before launching it.
    Returns True if all checks pass, False otherwise.
    """
    print(f"\n{'='*60}", flush=True)
    print("PRE-FLIGHT CHECK", flush=True)
    print(f"{'='*60}", flush=True)
    passed = True

    # 1. FL_SHARED_KEY must be set — proxcrypto.py will crash at import without it
    shared_key = os.environ.get("FL_SHARED_KEY", "")
    if not shared_key:
        print("❌ FL_SHARED_KEY is not set in the environment.", flush=True)
        print("   Generate one with:", flush=True)
        print("     python -c \"import os; print(os.urandom(32).hex())\"", flush=True)
        print("   Then set it before running:", flush=True)
        print("     Windows PowerShell:  $env:FL_SHARED_KEY = \"<hex_key>\"", flush=True)
        print("     Linux / macOS:       export FL_SHARED_KEY=\"<hex_key>\"", flush=True)
        passed = False
    else:
        try:
            key_bytes = bytes.fromhex(shared_key)
            if len(key_bytes) != 32:
                print(f"❌ FL_SHARED_KEY decodes to {len(key_bytes)} bytes — must be exactly 32.", flush=True)
                passed = False
            else:
                print(f"✔  FL_SHARED_KEY present and valid (32 bytes)", flush=True)
        except ValueError:
            print("❌ FL_SHARED_KEY is not a valid hex string.", flush=True)
            passed = False

    # 2. proxclient.py must exist alongside this script
    client_script = Path("proxclient.py")
    if not client_script.exists():
        print(f"❌ proxclient.py not found in: {Path.cwd()}", flush=True)
        passed = False
    else:
        print(f"✔  proxclient.py found", flush=True)

    # 3. Data directory must resolve correctly
    #    FL_DATA_DIR = output_dir.parent, client_name = output_dir.name
    #    proxclient.py joins them → output_dir
    fl_data_dir  = output_dir.parent.resolve()
    expected_data = fl_data_dir / client_name
    if not expected_data.exists():
        print(f"❌ Resolved data directory does not exist: {expected_data}", flush=True)
        passed = False
    else:
        print(f"✔  Data directory resolved correctly: {expected_data}", flush=True)

    # 4. train/ and test/ must both be present inside output_dir
    for split in ("train", "test"):
        split_path = output_dir / split
        if not split_path.exists():
            print(f"❌ Missing split folder: {split_path}", flush=True)
            passed = False
        else:
            print(f"✔  {split}/ folder present", flush=True)

    if passed:
        print("\n✅ All pre-flight checks passed.\n", flush=True)
    else:
        print("\n❌ Pre-flight checks FAILED — FL client will not be launched.\n", flush=True)

    return passed


# ==========================================================
# STEP 4 — launch proxclient.py
#
# FL_DATA_DIR is set to output_dir's PARENT.
# proxclient.py resolves:
#     os.path.join(FL_DATA_DIR, client_name)  →  output_dir
#     output_dir/train/<class>/               →  ImageFolder train  ✔
#     output_dir/test/<class>/                →  ImageFolder test   ✔
# ==========================================================
def launch_fl_client(client_name: str, output_dir: Path):
    # Run pre-flight checks before attempting to launch
    if not preflight_check(output_dir, client_name):
        sys.exit(1)

    fl_data_dir = str(output_dir.parent.resolve())
    env = os.environ.copy()
    env["FL_DATA_DIR"]      = fl_data_dir
    env["PYTHONUNBUFFERED"] = "1"
    # FL_SHARED_KEY is already in os.environ — carried over by .copy()
    # We do NOT log its value for security reasons.

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 4: Launching Federated Learning client", flush=True)
    print(f"   script      : proxclient.py", flush=True)
    print(f"   client_name : {client_name}", flush=True)
    print(f"   FL_DATA_DIR : {fl_data_dir}", flush=True)
    print(f"   data_dir    : {os.path.join(fl_data_dir, client_name)}", flush=True)
    print(f"   server      : localhost:8081", flush=True)
    print(f"   NOTE        : Make sure the server is already running!", flush=True)
    print(f"{'='*60}\n", flush=True)

    rc = run_subprocess(
        [sys.executable, "-u", "proxclient.py", client_name],
        env=env,
    )
    if rc != 0:
        print(f"\n❌ ERROR: proxclient.py exited with code {rc}", flush=True)
        print("   Common causes:", flush=True)
        print("   • FL server is not running on localhost:8081", flush=True)
        print("   • FL_SHARED_KEY does not match the server's key", flush=True)
        print("   • Server registration window has already closed", flush=True)
        sys.exit(1)


# ==========================================================
# MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Histopathology Filter + Restaining + FL Pipeline")
    parser.add_argument("--input_dir",  required=True,
                        help="Folder containing raw input images")
    parser.add_argument("--output_dir", required=True,
                        help="Where restained/classified images will be saved")
    parser.add_argument("--tmpdir", default="tmp_npz",
                        help="Temp folder for NPZ files")
    parser.add_argument("--clean_filtered", action="store_true",
                        help="Remove existing filtered_histo/ before running")
    args = parser.parse_args()

    input_dir    = Path(args.input_dir)
    output_dir   = Path(args.output_dir)
    tmp_dir      = Path(args.tmpdir)
    filtered_dir = Path("filtered_histo")

    # client_name = output folder name (e.g. "output")
    # FL_DATA_DIR = its parent  →  proxclient.py joins them → output_dir ✔
    client_name = output_dir.resolve().name

    if not input_dir.exists():
        print(f"❌ ERROR: {input_dir} does not exist.", flush=True)
        sys.exit(1)

    if args.clean_filtered and filtered_dir.exists():
        for p in filtered_dir.iterdir():
            try: p.unlink()
            except Exception: pass

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: filter ──────────────────────────────────────
    print("\nSTEP 1: Filtering non-histopathology images...\n", flush=True)
    filtered_images = preprocess_before_restaining(str(input_dir), str(filtered_dir))
    if not filtered_images:
        sys.exit(1)
    print(f"✔ {len(filtered_images)} histopathology images found.\n", flush=True)

    # ── STEP 2: restain ─────────────────────────────────────
    print("STEP 2: Restaining images...\n", flush=True)
    for img_path in filtered_images:
        try:
            process_single_image(Path(img_path), output_dir, tmp_dir)
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}", flush=True)
    print("\n🎉 Restaining complete!", flush=True)

    # ── STEP 3: classify + build train/test folders ─────────
    print("\nSTEP 3: Running predict_data_class.py...\n", flush=True)
    run_predict_script(output_dir)
    print("\n🎉 Classification complete!", flush=True)

    # ── verify folder structure before FL handoff ────────────
    if not verify_folder_structure(output_dir):
        print("\n❌ Folder structure check failed. Cannot launch FL client.", flush=True)
        print(f"   Expected: {output_dir}/train/<class>/<images>", flush=True)
        print(f"   Expected: {output_dir}/test/<class>/<images>", flush=True)
        sys.exit(1)
    print("\n✅ Folder structure verified.\n", flush=True)

    # ── STEP 4: launch FL client ─────────────────────────────
    launch_fl_client(client_name, output_dir)

    print("\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!\n", flush=True)


if __name__ == "__main__":
    main()