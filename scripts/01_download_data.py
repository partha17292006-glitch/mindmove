"""Step 1 — get BCI Competition IV 2a and build the per-subject epoch cache.

Order of attempts (WHY: never hang on a dead server):
  1. .mat files already on disk (Kaggle mirror or MOABB cache)  -> use them
  2. BNCI server reachable within 15 s                         -> MOABB (route A)
  3. Kaggle CLI available                                      -> download mirror (route B)
  4. otherwise print manual instructions.
Then epoch + cache all 9 subjects and verify 288 trials / session, 72 / class.
"""
import shutil
import subprocess
import sys
import urllib.request

import _bootstrap  # noqa: F401
import config as C
from mindmove import data

BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat"
KAGGLE_SLUG = "reader443/bci-competition-iv-dataset-2a-in-mat-format"


def have_local_files():
    return all(data.find_mat_file(s, sess) for s in C.SUBJECTS for sess in C.SESSIONS)


def bnci_reachable(timeout=15):
    try:
        req = urllib.request.Request(BNCI_URL, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  BNCI server not reachable ({type(exc).__name__}).")
        return False


def kaggle_download():
    if shutil.which("kaggle") is None:
        return False
    C.KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(C.KAGGLE_DIR), "--unzip"]
    print("  Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0 and have_local_files()


def main():
    print("[1/2] Locating raw data ...")
    if have_local_files():
        print("  Found all 18 .mat files on disk.")
    elif bnci_reachable():
        print("  Using MOABB (route A); files go to", C.MOABB_DIR)
    elif kaggle_download():
        print("  Downloaded Kaggle mirror (route B).")
    else:
        print("\nCould not obtain the data automatically. Manual route B:\n"
              f"  kaggle datasets download -d {KAGGLE_SLUG} -p data/raw/kaggle --unzip\n")
        sys.exit(1)

    print("[2/2] Epoching + caching + verifying ...")
    all_ok = True
    for s in C.SUBJECTS:
        data.build_cache(s)
        rep = data.verify_subject(s)
        for sess, r in rep.items():
            flag = "OK " if (r["ok"] and r["finite"]) else "BAD"
            all_ok &= r["ok"] and r["finite"]
            print(f"  A{s:02d}{sess}: {flag} shape={r['shape']} per_class={r['per_class']}")
    print("All subjects verified." if all_ok else "VERIFICATION FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
