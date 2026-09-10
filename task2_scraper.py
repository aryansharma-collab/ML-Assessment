"""
Task 2: Lexaloffle BBS Scraper
================================
Scrapes 100 PICO-8 games from the Lexaloffle BBS and exports to CSV.

Fields per game:
  1. Name
  2. Author
  3. Artwork URL
  4. Code (Lua source from .p8.png cart or page snippet)
  5. License
  6. Like count
  7. Description
  8. Top-5 comments
"""

import re
import csv
import json
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.lexaloffle.com"
BBS_URL = f"{BASE_URL}/bbs/"

# The BBS uses AJAX via lister.php. We replicate that call.
LISTER_URL = f"{BASE_URL}/bbs/lister.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": f"{BASE_URL}/bbs/?cat=7",
}

# Delay between requests to be polite
REQUEST_DELAY = 1.5  # seconds


def fetch_page(url, params=None):
    """Fetch a URL with retries and delay."""
    time.sleep(REQUEST_DELAY)
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            print(f"  [Retry {attempt+1}/3] Error fetching {url}: {e}")
            time.sleep(3)
    return None


def get_thread_ids_from_listing(num_pages=5):
    """
    Fetch thread IDs from the BBS listing pages.
    The lister.php endpoint returns HTML + JS with a `pdat` array.
    Each entry has: [post_id, thread_id, title, thumb_url, ...].
    We parse thread IDs from the noscript/JS fallback.
    """
    thread_ids = []

    for page_num in range(1, num_pages + 1):
        print(f"  Fetching listing page {page_num}...")
        params = {
            "cat": 7,
            "sub": 2,
            "page": page_num,
            "mode": "carts",
            "orderby": "ts",
        }
        html = fetch_page(LISTER_URL, params=params)
        if not html:
            print(f"  [!] Failed to fetch listing page {page_num}")
            continue

        # Extract pdat JavaScript array entries
        # Each pdat entry: ['post_id', thread_id, `title`, ...]
        pdat_match = re.findall(
            r"\['(\d+)',\s*(\d+),\s*`([^`]*)`",
            html
        )

        for post_id, tid, title in pdat_match:
            thread_ids.append({
                "post_id": post_id,
                "thread_id": tid,
                "title": title,
            })

        # Also try parsing thread links from HTML as fallback
        if not pdat_match:
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=re.compile(r"\?tid=\d+")):
                tid_m = re.search(r"tid=(\d+)", link["href"])
                if tid_m:
                    title = link.get_text(strip=True)
                    thread_ids.append({
                        "post_id": "",
                        "thread_id": tid_m.group(1),
                        "title": title,
                    })

        if len(thread_ids) >= 100:
            break

    # Deduplicate by thread_id
    seen = set()
    unique = []
    for t in thread_ids:
        if t["thread_id"] not in seen:
            seen.add(t["thread_id"])
            unique.append(t)

    return unique[:100]


def scrape_thread(thread_id):
    """
    Scrape a single PICO-8 game thread page.
    Returns a dict with all 8 fields.
    """
    url = f"{BBS_URL}?tid={thread_id}"
    html = fetch_page(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    game = {
        "name": "",
        "author": "",
        "artwork_url": "",
        "code": "",
        "license": "Not specified",
        "like_count": 0,
        "description": "",
        "top_5_comments": "",
    }

    # --- Name ---
    title_el = soup.find("span", class_="post_title2") or soup.find("title")
    if title_el:
        game["name"] = title_el.get_text(strip=True)
        # Clean up "Lexaloffle BBS ::" prefix from <title>
        if "Lexaloffle BBS" in game["name"]:
            game["name"] = game["name"].replace("Lexaloffle BBS :: ", "").strip()

    # --- Author ---
    # Author is in a link like /bbs/?uid=XXXX. Find the first one with actual text.
    for author_link in soup.find_all("a", href=re.compile(r"/bbs/\?uid=\d+")):
        text = author_link.get_text(strip=True)
        if text and not text.startswith("@") and text != "More cartridges":
            game["author"] = text
            break

    # --- Artwork URL ---
    # The cart thumbnail: /bbs/thumbs/pico8_XXXXX.png
    thumb_img = soup.find("img", src=re.compile(r"/bbs/thumbs/pico8_"))
    if thumb_img:
        src = thumb_img.get("src", "")
        game["artwork_url"] = BASE_URL + src if src.startswith("/") else src

    # Also try the dormant player label image
    if not game["artwork_url"]:
        dormant_img = soup.find("div", id=re.compile(r"dormant_label_container"))
        if dormant_img:
            img = dormant_img.find("img")
            if img:
                src = img.get("src", "")
                game["artwork_url"] = BASE_URL + src if src.startswith("/") else src

    # --- License ---
    # License info appears near the cart info bar, e.g. "No License" or "CC-BY-NC-SA 4.0"
    if "CC-BY-NC-SA" in html:
        game["license"] = "CC-BY-NC-SA 4.0"
    elif "CC-BY 4.0" in html:
        game["license"] = "CC-BY 4.0"
    elif "No License" in html:
        game["license"] = "No License"
    else:
        # Try a broader regex for any license text near the cart info
        license_match = re.search(r'>\s*((?:No License|CC[^<]{3,30}))</(?:a|span|div)', html)
        if license_match:
            game["license"] = license_match.group(1).strip()
        else:
            game["license"] = "Not specified"

    # --- Like count ---
    # Likes are in the rate div: <div style="...">15</div> next to like icon
    like_match = re.search(
        r'class="i_rate_\d+_like".*?</div>\s*<div[^>]*>(\d+)</div>',
        html, re.DOTALL
    )
    if like_match:
        game["like_count"] = int(like_match.group(1))

    # --- Code ---
    # Try to extract code from hidden cartsrc div or the .p8.png URL
    cartsrc_div = soup.find("div", id=re.compile(r"cartsrc_"))
    if cartsrc_div:
        code_text = cartsrc_div.get_text(strip=True)
        if code_text:
            game["code"] = code_text[:5000]  # Limit to 5000 chars

    # If no inline code, try to get cart URL for download
    if not game["code"]:
        cart_link = soup.find("a", href=re.compile(r"/bbs/cposts/.*\.p8\.png"))
        if cart_link:
            cart_url = cart_link["href"]
            if cart_url.startswith("/"):
                cart_url = BASE_URL + cart_url
            game["code"] = f"[Cart: {cart_url}]"

    # --- Description ---
    # The post body is typically in a <div> after the cart player
    # Look for the main post content
    post_body = ""

    # Try finding the post description - it's usually in a div with specific styling
    # after the cart player section
    all_text_divs = soup.find_all("div", style=re.compile(r"padding.*display"))
    for div in all_text_divs:
        text = div.get_text(strip=True)
        # Filter out navigation/menu text and get actual description
        if len(text) > 50 and "BBS" not in text[:20] and "PICO-8" not in text[:20]:
            if "Cart" not in text[:10] and "Search:" not in text:
                post_body = text
                break

    # Alternative: Look for paragraphs in the main content area
    if not post_body:
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 30:
                post_body += text + " "

    game["description"] = post_body[:2000].strip() if post_body else "No description available"

    # --- Top 5 Comments ---
    # Comments are in divs with class containing "thread_preview" or reply divs
    comments = []
    # Look for reply posts - they typically follow the main post
    reply_divs = soup.find_all("div", class_=re.compile(r"thread_preview|post_body"))
    for div in reply_divs[:5]:
        comment_text = div.get_text(strip=True)
        if comment_text and len(comment_text) > 5:
            comments.append(comment_text[:500])

    # Alternative: look for all post content after the first one
    if not comments:
        all_posts = soup.find_all("div", id=re.compile(r"pdat_\d+"))
        for post in all_posts[1:6]:  # Skip first (main post), take up to 5
            text = post.get_text(strip=True)
            if text:
                comments.append(text[:500])

    game["top_5_comments"] = " ||| ".join(comments) if comments else "No comments"

    return game


def main():
    print("=" * 60)
    print("  PICO-8 BBS Scraper - Task 2")
    print("=" * 60)

    # Step 1: Get thread IDs from listing pages
    print("\n[1/2] Fetching thread listings...")
    threads = get_thread_ids_from_listing(num_pages=7)
    print(f"  Found {len(threads)} unique threads.")

    if len(threads) < 100:
        print(f"  [!] Warning: Only found {len(threads)} threads (target: 100).")
        print("      Trying additional pages...")
        more_threads = get_thread_ids_from_listing(num_pages=15)
        # Merge
        seen_tids = {t["thread_id"] for t in threads}
        for t in more_threads:
            if t["thread_id"] not in seen_tids:
                threads.append(t)
                seen_tids.add(t["thread_id"])
        threads = threads[:100]

    # Step 2: Scrape each thread
    print(f"\n[2/2] Scraping {len(threads)} game threads...")
    games = []
    for i, thread in enumerate(threads):
        tid = thread["thread_id"]
        title = thread.get("title", "Unknown")
        print(f"  [{i+1}/{len(threads)}] Scraping: {title} (tid={tid})")

        game_data = scrape_thread(tid)
        if game_data:
            # Use listing title if page title wasn't found
            if not game_data["name"] and title:
                game_data["name"] = title
            games.append(game_data)
        else:
            print(f"    [!] Failed to scrape thread {tid}")

    # Step 3: Export to CSV
    output_file = "pico8_games.csv"
    print(f"\n  Writing {len(games)} games to {output_file}...")

    fieldnames = [
        "name", "author", "artwork_url", "code",
        "license", "like_count", "description", "top_5_comments"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(games)

    print(f"\n[✓] Saved {len(games)} games to '{output_file}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
