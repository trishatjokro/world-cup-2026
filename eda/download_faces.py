"""Download EA FC face images locally so the app serves them from its own
origin (third-party CDNs like cdn.sofifa.net get blocked by browser privacy
filters). Adds a `face_file` column (relative path) to data/players.csv.
"""
import os
import concurrent.futures as cf

import pandas as pd
import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES = os.path.join(HERE, "data", "faces")
os.makedirs(FACES, exist_ok=True)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def fid(url: str) -> str:
    # https://cdn.sofifa.net/players/204/485/23_120.png -> 204485_23
    parts = url.split("/players/")[1].split("/")
    return f"{parts[0]}{parts[1]}_{parts[2].split('_')[0]}"


def fetch(url: str):
    name = fid(url) + ".png"
    dest = os.path.join(FACES, name)
    rel = os.path.join("data", "faces", name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return url, rel
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://sofifa.com/"},
                         timeout=20)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            with open(dest, "wb") as f:
                f.write(r.content)
            return url, rel
    except Exception:
        pass
    return url, ""


def main():
    p = pd.read_csv(os.path.join(HERE, "data", "players.csv"))
    todo = p[p.has_photo & p.face_url.notna()]["face_url"].unique().tolist()
    print(f"downloading {len(todo)} face images ...")
    mapping = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, (url, rel) in enumerate(ex.map(fetch, todo), 1):
            mapping[url] = rel
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    p["face_file"] = p["face_url"].map(lambda u: mapping.get(u, "") if isinstance(u, str) else "")
    ok = (p["face_file"] != "").sum()
    p.to_csv(os.path.join(HERE, "data", "players.csv"), index=False)
    print(f"done: {ok}/{len(p)} players have a local face file; "
          f"wrote data/players.csv (+face_file) and {len(os.listdir(FACES))} images")


if __name__ == "__main__":
    main()
