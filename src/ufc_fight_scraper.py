import logging, time, requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ufcstats")

# HTTP session with retries
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Safari/537.36"})
    retry = Retry(total=5, backoff_factor=0.8, status_forcelist=(429,500,502,503,504), allowed_methods={"GET","HEAD"})
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

SESSION = make_session()

# Base URL and fetch helpers
BASE_EVENTS = "http://ufcstats.com/statistics/events/completed?page="

def get_page_html(page_number, sleep=0.2):
    url = BASE_EVENTS + str(page_number)
    r = SESSION.get(url, timeout=20)
    r.raise_for_status()
    time.sleep(sleep)
    return BeautifulSoup(r.text, "html.parser")

def get_event_links():
    all_events, page = [], 1
    while True:
        soup = get_page_html(page)
        links = []
        for a in soup.select('a[href*="/event-details/"]'):
            href = a.get("href")
            name = a.get_text(strip=True)
            if href and name:
                links.append((name, href))
        # dedup the page while preserving order
        seen, uniq = set(), []
        for name, href in links:
            if href not in seen:
                seen.add(href)
                uniq.append((name, href))
        if not uniq:
            break
        all_events.extend(uniq)
        log.info(f"Page {page}: {len(uniq)} events")
        page += 1
    # final unique list
    out, seen = [], set()
    for name, href in all_events:
        if href not in seen:
            seen.add(href)
            out.append({"event_name": name, "event_url": href})
    log.info(f"Total events: {len(out)}")
    return out

def parse_event_bouts(event_name, event_url, sleep=0.2):
    r = SESSION.get(event_url, timeout=20)
    r.raise_for_status()
    time.sleep(sleep)
    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("tbody tr")
    fights = []
    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue
        fighter_cell = tds[1]
        names = [p.get_text(strip=True) for p in fighter_cell.find_all("p")][:2]
        bout_a = tr.select_one('a[href*="/fight-details/"]')
        fights.append({
            "event": event_name,
            "event_url": event_url,
            "fighter_red": names[0] if len(names) > 0 else None,
            "fighter_blue": names[1] if len(names) > 1 else None,
            "result": tds[0].get_text(strip=True),
            "method": tds[7].get_text(strip=True) if len(tds) > 7 else None,
            "round":  tds[8].get_text(strip=True) if len(tds) > 8 else None,
            "time":   tds[9].get_text(strip=True) if len(tds) > 9 else None,
            "bout_url": bout_a["href"] if bout_a else None,
        })
    return fights

# Crawl all events, gather all fights, save one CSV
def main():
    events = get_event_links()
    all_fights = []
    for i, ev in enumerate(events, 1):
        log.info(f"[{i}/{len(events)}] {ev['event_name']}")
        fights = parse_event_bouts(ev["event_name"], ev["event_url"])
        log.info(f"  parsed {len(fights)} fights")
        all_fights.extend(fights)

    df = pd.DataFrame(all_fights)
    out_dir = Path.home() / "projects" / "ufc"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "ufc_fights_all.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} fights to {outfile}")

if __name__ == "__main__":
    main()
