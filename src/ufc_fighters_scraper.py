import logging
import time
import re
import string
import os
from datetime import datetime
from typing import Dict, Optional, List

import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ufcstats_fighters")

# HTTP session

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Safari/537.36"})
    retry = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


S = make_session()

# Converters

def height_to_inches(txt: Optional[str]) -> Optional[int]:
    if not txt:
        return None
    m = re.search(r"(\d+)\s*'\s*(\d+)", txt)
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def reach_to_inches(txt: Optional[str]) -> Optional[float]:
    if not txt:
        return None
    m = re.search(r"(\d+(\.\d+)?)", txt)
    return float(m.group(1)) if m else None


def weight_to_lbs(txt: Optional[str]) -> Optional[float]:
    if not txt:
        return None
    m = re.search(r"(\d+(\.\d+)?)", txt)
    return float(m.group(1)) if m else None


def pct_to_float(txt: Optional[str]) -> Optional[float]:
    if not txt:
        return None
    m = re.search(r"(\d+(\.\d+)?)\s*%", txt)
    return float(m.group(1)) / 100.0 if m else None


def num_or_none(txt: Optional[str]) -> Optional[float]:
    if txt is None:
        return None
    try:
        return float(txt)
    except ValueError:
        return None



# Small HTML helpers

def read_li_label_values(ul) -> Dict[str, str]:
    """
    Parse <ul> where each <li> is like:
      <li><i>Label:</i> value</li>
    Returns {label_lowercase_no_colon: raw_value_string}
    """
    out = {}
    for li in ul.select("li"):
        i_tag = li.find("i")
        if not i_tag:
            continue
        label = i_tag.get_text(strip=True)
        parts = list(li.stripped_strings)
        if parts and parts[0] == label:
            parts = parts[1:]
        value = " ".join(parts).strip()
        key = label.strip(": ").lower()
        out[key] = value
    return out



# Fighter profile parser

def parse_fighter_profile(fighter_url: str) -> Dict[str, Optional[float]]:
    # Force http for reliability on your network
    fighter_url = fighter_url.replace("https://", "http://")

    r = S.get(fighter_url, timeout=20)
    r.raise_for_status()
    time.sleep(0.15)
    soup = BeautifulSoup(r.text, "html.parser")

    # Career stats live in two <ul> under the wrapper
    stats_raw = {}
    wrap = soup.find("div", class_="b-list__info-box-left clearfix")
    if wrap:
        for ul in wrap.select("ul.b-list__box-list"):
            stats_raw.update(read_li_label_values(ul))

    # Bio list (height, reach, stance, weight, dob)
    bio_raw = {}
    for ul in soup.select("ul.b-list__box-list"):
        if ul.find("i", string=lambda t: t and t.strip().lower() in {"height:", "reach:", "dob:", "stance:", "weight:"}):
            bio_raw.update(read_li_label_values(ul))
            break

    # Name
    name_tag = soup.select_one("span.b-content__title-highlight")
    name = name_tag.get_text(strip=True) if name_tag else None

    # Normalize
    out = {
        "fighter_url": fighter_url,
        "name": name,
        "height_in": height_to_inches(bio_raw.get("height")),
        "reach_in": reach_to_inches(bio_raw.get("reach")),
        "weight_lbs": weight_to_lbs(bio_raw.get("weight")),
        "stance": bio_raw.get("stance") or None,
        "dob": None,
        "slpm": num_or_none(stats_raw.get("slpm")),
        "str_acc": pct_to_float(stats_raw.get("str. acc.")),
        "sapm": num_or_none(stats_raw.get("sapm")),
        "str_def": pct_to_float(stats_raw.get("str. def")),
        "td_avg": num_or_none(stats_raw.get("td avg.")),
        "td_acc": pct_to_float(stats_raw.get("td acc.")),
        "td_def": pct_to_float(stats_raw.get("td def.")),
        "sub_avg": num_or_none(stats_raw.get("sub. avg.")),
    }

    # DOB to ISO if possible
    dob_raw = bio_raw.get("dob")
    if dob_raw:
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                out["dob"] = datetime.strptime(dob_raw, fmt).date().isoformat()
                break
            except ValueError:
                continue
        if out["dob"] is None:
            out["dob"] = dob_raw

    return out


# Fighters index

def fetch_fighter_list_for_letter(ch: str) -> List[str]:
    """
    Returns a list of fighter profile URLs for one letter.
    """
    url = f"http://ufcstats.com/statistics/fighters?char={ch}&page=all"
    r = S.get(url, timeout=20)
    r.raise_for_status()
    time.sleep(0.1)
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.select('a[href*="/fighter-details/"]'):
        href = a.get("href")
        if href:
            links.append(href)

    # Dedup within letter while keeping order
    seen = set()
    uniq = []
    for href in links:
        if href not in seen:
            seen.add(href)
            uniq.append(href)

    log.info(f"Letter {ch}: {len(uniq)} fighters")
    return uniq


def collect_all_unique_profile_urls() -> List[str]:
    letters = list(string.ascii_lowercase) + ["*"]  # "*" often covers numeric or special
    all_links = []
    for ch in letters:
        try:
            all_links.extend(fetch_fighter_list_for_letter(ch))
        except requests.RequestException as e:
            log.warning(f"Skipping letter {ch} due to network error: {e}")

    # Final unique set across all letters
    unique_links = sorted(set(all_links))
    log.info(f"Found {len(unique_links)} unique fighter URLs")
    return unique_links

# Resume support

def load_seen_from_csv(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["fighter_url"])
        return set(df["fighter_url"].dropna().astype(str).tolist())
    except Exception:
        return set()


def append_rows_to_csv(rows: List[Dict], csv_path: str) -> None:
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

# Main

def main():
    out_csv = os.path.expanduser("~/projects/ufc/ufc_fighters_master.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)


    # Step 1: collect all unique profile URLs
    urls = collect_all_unique_profile_urls()

    # Step 2: resume support
    seen = load_seen_from_csv(out_csv)
    if seen:
        log.info(f"Resume mode: {len(seen)} already saved")

    # Step 3: scrape each profile once
    total = len(urls)
    batch, batch_size = [], 50
    scraped = len(seen)

    for i, url in enumerate(urls, 1):
        if url in seen:
            continue
        try:
            data = parse_fighter_profile(url)
            batch.append(data)
            scraped += 1
        except requests.RequestException as e:
            log.warning(f"Failed {url}: {e}")
            continue

        if len(batch) >= batch_size:
            append_rows_to_csv(batch, out_csv)
            log.info(f"Scraped {scraped}/{total} fighters")
            batch.clear()

    # flush remaining
    if batch:
        append_rows_to_csv(batch, out_csv)
        log.info(f"Scraped {scraped}/{total} fighters")

    # Final dedup pass on disk to be safe
    df = pd.read_csv(out_csv)
    before = len(df)
    df = df.drop_duplicates(subset=["fighter_url"]).reset_index(drop=True)
    after = len(df)
    if after != before:
        df.to_csv(out_csv, index=False)
    print(f"Saved {after} unique fighters to {out_csv}")


if __name__ == "__main__":
    main()
