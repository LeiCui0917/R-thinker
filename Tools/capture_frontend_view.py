import argparse
import os
import time
from datetime import datetime


def _default_selector(mode: str) -> str:
    if mode == 'chess':
        return '#board'
    if mode == 'maze':
        return '#mazeCanvas'
    raise ValueError(f'Unsupported mode: {mode}')


def _build_output_path(path: str, index: int, multi: bool) -> str:
    root, ext = os.path.splitext(path)
    ext = ext or '.png'
    if not multi:
        return root + ext
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{root}_{stamp}_{index:03d}{ext}'


def _wait_target_visible(page, selector: str, timeout_ms: int) -> None:
    page.locator(selector).first.wait_for(state='visible', timeout=timeout_ms)


def _capture_once(page, selector: str, out_path: str) -> None:
    locator = page.locator(selector).first
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    locator.screenshot(path=out_path, animations='disabled')


def main() -> int:
    ap = argparse.ArgumentParser(description='Capture exact board/maze screenshot from running frontend page.')
    ap.add_argument('--url', required=True, help='Frontend URL, e.g. http://127.0.0.1:5000')
    ap.add_argument('--mode', choices=['chess', 'maze'], required=True, help='Capture target preset')
    ap.add_argument('--selector', default='', help='Optional CSS selector override')
    ap.add_argument('--out', default='', help='Output png path (default: Tools/captures/<mode>.png)')
    ap.add_argument('--wait-ms', type=int, default=15000, help='Wait timeout for page/element in milliseconds')
    ap.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    ap.add_argument('--interval', type=float, default=0.0, help='If >0, capture repeatedly every N seconds')
    ap.add_argument('--count', type=int, default=1, help='Capture count when --interval > 0 (0 means infinite)')
    args = ap.parse_args()

    selector = args.selector.strip() or _default_selector(args.mode)
    out_path = args.out.strip() or os.path.join('Tools', 'captures', f'{args.mode}.png')

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print('Playwright is required. Install with: pip install playwright ; playwright install chromium')
        print(f'Import error: {exc}')
        return 2

    multi = args.interval > 0
    max_count = int(args.count)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=bool(args.headless))
        try:
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            page.goto(args.url, wait_until='domcontentloaded', timeout=args.wait_ms)
            _wait_target_visible(page, selector, timeout_ms=args.wait_ms)

            i = 0
            while True:
                i += 1
                shot_path = _build_output_path(out_path, i, multi)
                try:
                    _capture_once(page, selector, shot_path)
                    print(f'Saved: {shot_path}')
                except Exception as exc:
                    print(f'Capture failed for selector "{selector}": {exc}')
                    return 1

                if not multi:
                    break
                if max_count > 0 and i >= max_count:
                    break
                time.sleep(max(0.0, float(args.interval)))
        finally:
            browser.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
