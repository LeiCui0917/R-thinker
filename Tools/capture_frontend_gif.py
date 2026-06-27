import argparse
import os
import time
from io import BytesIO


def _default_selector(mode: str) -> str:
    if mode == 'chess':
        return '#board'
    if mode == 'maze':
        return '#mazeCanvas'
    raise ValueError(f'Unsupported mode: {mode}')


def _wait_target_visible(page, selector: str, timeout_ms: int) -> None:
    page.locator(selector).first.wait_for(state='visible', timeout=timeout_ms)


def _capture_frame(page, selector: str):
    return page.locator(selector).first.screenshot(type='png', animations='disabled')


def main() -> int:
    ap = argparse.ArgumentParser(description='Capture running frontend board/canvas frames and generate a GIF.')
    ap.add_argument('--url', required=True, help='Frontend URL, e.g. http://127.0.0.1:5000')
    ap.add_argument('--mode', choices=['chess', 'maze'], required=True, help='Capture target preset')
    ap.add_argument('--selector', default='', help='Optional CSS selector override')
    ap.add_argument('--out', default='', help='Output gif path (default: Tools/captures/<mode>.gif)')
    ap.add_argument('--wait-ms', type=int, default=15000, help='Wait timeout for page/element in milliseconds')
    ap.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    ap.add_argument('--interval', type=float, default=0.2, help='Seconds between captured frames')
    ap.add_argument('--count', type=int, default=30, help='Number of frames to capture')
    ap.add_argument('--loop', type=int, default=0, help='GIF loop count (0 means infinite)')
    ap.add_argument('--frame-ms', type=int, default=0, help='GIF frame duration in ms (0 means auto from interval)')
    args = ap.parse_args()

    if args.count <= 0:
        print('--count must be > 0')
        return 2
    if args.interval < 0:
        print('--interval must be >= 0')
        return 2

    selector = args.selector.strip() or _default_selector(args.mode)
    out_path = args.out.strip() or os.path.join('Tools', 'captures', f'{args.mode}.gif')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    try:
        from PIL import Image
    except Exception as exc:
        print('Pillow is required. Install with: pip install pillow')
        print(f'Import error: {exc}')
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print('Playwright is required. Install with: pip install playwright ; playwright install chromium')
        print(f'Import error: {exc}')
        return 2

    frame_duration_ms = int(args.frame_ms) if int(args.frame_ms) > 0 else max(1, int(args.interval * 1000))
    frames = []
    frame_count = int(args.count)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=bool(args.headless))
        try:
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            page.goto(args.url, wait_until='domcontentloaded', timeout=args.wait_ms)
            _wait_target_visible(page, selector, timeout_ms=args.wait_ms)

            for idx in range(frame_count):
                try:
                    png_bytes = _capture_frame(page, selector)
                    frame = Image.open(BytesIO(png_bytes)).convert('P', palette=Image.Palette.ADAPTIVE)
                    frames.append(frame)
                    print(f'Captured frame {idx + 1}/{frame_count}')
                except Exception as exc:
                    print(f'Capture failed for selector "{selector}": {exc}')
                    return 1

                if idx < frame_count - 1 and args.interval > 0:
                    time.sleep(float(args.interval))
        finally:
            browser.close()

    if not frames:
        print('No frames captured, GIF was not created.')
        return 1

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=int(args.loop),
        optimize=False,
    )
    print(f'Saved GIF: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())