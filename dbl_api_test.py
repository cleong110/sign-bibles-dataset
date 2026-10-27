"""
Quick standalone test of the new DBL REST API, using "Old Style" auth
(email + key derived from email/password). Run this locally — it never
sends your credentials anywhere but thedigitalbiblelibrary.org.

Usage:
    python dbl_api_test.py

It'll prompt for email/password interactively (not echoed), fetch a key,
then hit open_access_entries_list and print the raw JSON for the first
few entries so we can inspect the schema (especially the "type" field
for sign language video entries, which isn't documented as "video" —
could be "audio", could be something else).
"""

import getpass
import json

import requests

BASE = "https://thedigitalbiblelibrary.org"


def get_key(email: str, password: str) -> str:
    resp = requests.get(f"{BASE}/api/user_token", params={"email": email, "password": password})
    resp.raise_for_status()
    data = resp.json()
    # Docs don't show the exact response shape for this endpoint, so we
    # print it raw if "key" isn't where we expect.
    if "key" in data:
        return data["key"]
    print("Unexpected response shape from /api/user_token:")
    print(json.dumps(data, indent=2))
    raise SystemExit(1)


def api_get(path: str, email: str, key: str, **extra_params):
    params = {"email": email, "key": key, **extra_params}
    resp = requests.get(f"{BASE}{path}", params=params)
    print(f"GET {path} -> {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


def main():
    email = input("DBL email: ").strip()
    password = getpass.getpass("DBL password (not echoed): ")

    print("\nFetching key...")
    key = get_key(email, password)
    print("Got a key. (Not printing it.)\n")

    print("Fetching open_access_entries_list...")
    data = api_get("/api/open_access_entries_list", email, key)

    entries = data.get("entries", data if isinstance(data, list) else [])
    print(f"\nGot {len(entries)} entries. Showing first 5 raw, full, unfiltered:\n")
    for entry in entries[:5]:
        print(json.dumps(entry, indent=2))
        print("---")

    # Specifically surface what "type" values actually show up, since the
    # docs only mention text|audio and our sign language entries might be
    # filed under something unexpected.
    types_seen = {e.get("type") for e in entries if isinstance(e, dict)}
    print(f"\nDistinct 'type' values seen across all {len(entries)} entries: {types_seen}")

    # Try to spot anything that looks sign-language-related by name/language,
    # so we have a concrete entry id to test the revision/license endpoints on.
    candidates = [
        e
        for e in entries
        if isinstance(e, dict)
        and ("sign" in json.dumps(e).lower())
    ]
    print(f"\nEntries with 'sign' anywhere in their fields: {len(candidates)}")
    for c in candidates[:5]:
        print(json.dumps(c, indent=2))
        print("---")


if __name__ == "__main__":
    main()
