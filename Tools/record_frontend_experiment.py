"""
实验一键录制工具（前端棋盘/迷宫）

功能：
1) 自动等待：打开页面后，先等待目标元素可见，再按 --startup-wait 预热等待。
2) 时间戳命名：默认输出前缀为 <mode>_<YYYYMMDD_HHMMSS>。
3) 双输出：同一次录制同时产出 PNG 序列和 GIF。

运行前准备：
1) 启动你的前端页面（例如 chess 在 5000 端口，maze 在 5001 端口）。
2) 安装依赖：
    pip install playwright pillow
3) 首次安装浏览器驱动：
    playwright install chromium

最常用命令：
1) 录制 chess（默认输出到 Tools/captures）：
    python Tools/record_frontend_experiment.py --url http://127.0.0.1:5000 --mode chess --count 60 --interval 0.2

2) 录制 maze（启动后额外等待 3 秒再开始）：
    python Tools/record_frontend_experiment.py --url http://127.0.0.1:5001 --mode maze --startup-wait 3 --count 80 --interval 0.15

3) 指定输出目录和前缀：
    python Tools/record_frontend_experiment.py --url http://127.0.0.1:5000 --mode chess --out-dir Tools/captures --prefix exp_demo

输出结构示例：
Tools/captures/chess_20260321_142530/
  ├─ chess_20260321_142530.gif
  └─ png/
        ├─ chess_20260321_142530_0001.png
        ├─ chess_20260321_142530_0002.png
        └─ ...
"""

import argparse
import os
import time
from datetime import datetime
from io import BytesIO


def _default_selector(mode: str) -> str:
    if mode == 'chess':
        return '#board'
    if mode == 'maze':
        return '#mazeCanvas'
    raise ValueError(f'Unsupported mode: {mode}')


def _wait_target_visible(page, selector: str, timeout_ms: int) -> None:
    page.locator(selector).first.wait_for(state='visible', timeout=timeout_ms)


def _capture_png_bytes(page, selector: str) -> bytes:
    return page.locator(selector).first.screenshot(type='png', animations='disabled')


def main() -> int:
    ap = argparse.ArgumentParser(
        description='One-click frontend experiment recorder: auto-wait, timestamp naming, PNG sequence + GIF output.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python Tools/record_frontend_experiment.py --url http://127.0.0.1:5000 --mode chess --count 60 --interval 0.2\n'
            '  python Tools/record_frontend_experiment.py --url http://127.0.0.1:5001 --mode maze --startup-wait 3 --count 80 --interval 0.15\n'
            '  python Tools/record_frontend_experiment.py --url http://127.0.0.1:5000 --mode chess --prefix exp_demo --out-dir Tools/captures'
        ),
    )
    ap.add_argument('--url', required=True, help='Frontend URL, e.g. http://127.0.0.1:5000')
    ap.add_argument('--mode', choices=['chess', 'maze'], required=True, help='Capture target preset')
    ap.add_argument('--selector', default='', help='Optional CSS selector override')
    ap.add_argument('--out-dir', default=os.path.join('Tools', 'captures'), help='Output directory root')
    ap.add_argument('--prefix', default='', help='Output name prefix (default: <mode>_<timestamp>)')
    ap.add_argument('--wait-ms', type=int, default=15000, help='Wait timeout for page/element in milliseconds')
    ap.add_argument('--startup-wait', type=float, default=2.0, help='Extra wait after page load before recording (seconds)')
    ap.add_argument('--interval', type=float, default=0.2, help='Seconds between frames')
    ap.add_argument('--count', type=int, default=30, help='Number of frames to capture')
    ap.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    ap.add_argument('--loop', type=int, default=0, help='GIF loop count (0 means infinite)')
    ap.add_argument('--frame-ms', type=int, default=0, help='GIF frame duration in ms (0 means auto from interval)')
    args = ap.parse_args()

    if args.count <= 0:
        print('--count must be > 0')
        return 2
    if args.interval < 0:
        print('--interval must be >= 0')
        return 2
    if args.startup_wait < 0:
        print('--startup-wait must be >= 0')
        return 2

    selector = args.selector.strip() or _default_selector(args.mode)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = args.prefix.strip() or f'{args.mode}_{stamp}'

    run_dir = os.path.join(args.out_dir, prefix)
    png_dir = os.path.join(run_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    gif_path = os.path.join(run_dir, f'{prefix}.gif')

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

    print(f'Recording URL: {args.url}')
    print(f'Selector: {selector}')
    print(f'Output run dir: {run_dir}')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=bool(args.headless))
        try:
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            page.goto(args.url, wait_until='domcontentloaded', timeout=args.wait_ms)

            _wait_target_visible(page, selector, timeout_ms=args.wait_ms)
            if args.startup_wait > 0:
                print(f'Warmup wait: {args.startup_wait:.2f}s')
                time.sleep(float(args.startup_wait))

            for idx in range(frame_count):
                try:
                    png_bytes = _capture_png_bytes(page, selector)
                    png_path = os.path.join(png_dir, f'{prefix}_{idx + 1:04d}.png')
                    with open(png_path, 'wb') as f:
                        f.write(png_bytes)

                    frame = Image.open(BytesIO(png_bytes)).convert('P', palette=Image.Palette.ADAPTIVE)
                    frames.append(frame)
                    print(f'Captured {idx + 1}/{frame_count}: {png_path}')
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
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=int(args.loop),
        optimize=False,
    )

    print(f'Saved GIF: {gif_path}')
    print(f'Saved PNG sequence dir: {png_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())