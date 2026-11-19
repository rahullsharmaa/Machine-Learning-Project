"""Download the challenge dataset folder from Google Drive (via the Lovable connector gateway) into a
local directory, preserving the folder structure. Usage: fetch_dataset.py <folder_id> <dest_dir>"""
import os, sys, time, json, urllib.request, urllib.parse

GW = "https://connector-gateway.lovable.dev/google_drive"
HDR = {"Authorization": f"Bearer {os.environ['LOVABLE_API_KEY']}", "X-Connection-Api-Key": os.environ["GOOGLE_DRIVE_API_KEY"]}


def api(path, params=None, stream_to=None):
    url = GW + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=600) as r:
                if stream_to is None:
                    return json.loads(r.read().decode())
                n = 0
                with open(stream_to, "wb") as f:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        f.write(b)
                        n += len(b)
                return n
        except Exception as e:  # noqa
            print(f"  retry {attempt+1} for {path}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {path}")


def walk(folder_id, dest):
    os.makedirs(dest, exist_ok=True)
    page = None
    while True:
        params = {"q": f"'{folder_id}' in parents and trashed = false", "fields": "nextPageToken,files(id,name,mimeType,size)",
                  "pageSize": 1000, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}
        if page:
            params["pageToken"] = page
        res = api("/drive/v3/files", params)
        for f in res.get("files", []):
            path = os.path.join(dest, f["name"])
            if f["mimeType"] == "application/vnd.google-apps.folder":
                walk(f["id"], path)
            else:
                size = int(f.get("size", 0))
                if os.path.exists(path) and os.path.getsize(path) == size:
                    print(f"  have {path} ({size:,} B)", flush=True)
                    continue
                t = time.time()
                n = api(f"/drive/v3/files/{f['id']}", {"alt": "media", "supportsAllDrives": "true"}, stream_to=path)
                print(f"  got {path} ({n:,} B in {time.time()-t:.0f}s)", flush=True)
                if size and n != size:
                    raise RuntimeError(f"size mismatch for {path}: {n} != {size}")
        page = res.get("nextPageToken")
        if not page:
            break


if __name__ == "__main__":
    walk(sys.argv[1], sys.argv[2])
    print("done")
