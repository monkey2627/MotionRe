"""Read public research pages and save a reproducible text/URL receipt."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def fetch(url, output_dir, proxy, link_filter):
    response = requests.get(
        url, timeout=35,
        proxies={"http": proxy, "https": proxy} if proxy else None,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for item in soup(["script", "style", "nav"]):
        item.decompose()
    records = soup.select("li.arxiv-result")
    if records:
        content = "\n\n".join(item.get_text(" ", strip=True) for item in records)
        overview = []
        for item in records:
            title = item.select_one("p.title")
            link = item.select_one("p.list-title a")
            overview.append({"title": title.get_text(" ", strip=True),
                             "url": link.get("href")})
    else:
        content = soup.get_text("\n", strip=True)
        overview = content[:1800]
    key = hashlib.sha256(url.encode()).hexdigest()[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (key + ".txt")
    target.write_text("URL: " + url + "\n\n" + content, encoding="utf-8")
    result = {"url": url, "status": response.status_code,
              "saved": str(target), "content": overview}
    if link_filter:
        result["matching_links"] = [
            {"title": item.get_text(" ", strip=True), "href": item.get("href")}
            for item in soup.select("a[href]")
            if link_filter.lower() in item.get_text(" ", strip=True).lower()
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--link-filter", default="")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "sources")
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, url, args.output_dir, args.proxy,
                               args.link_filter): url
                   for url in args.urls}
        for future in concurrent.futures.as_completed(futures):
            try:
                print(json.dumps(future.result(), ensure_ascii=True))
            except Exception as exc:
                print(json.dumps({"url": futures[future], "error": str(exc)}))


if __name__ == "__main__":
    main()
