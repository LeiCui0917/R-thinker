"""LLM HTTP 请求公共工具（极简重试版）。

职责：
- 统一发送 JSON POST 请求；
- 对瞬时错误做有限重试（带指数退避）；
- 返回统一三元组，供上层（LLMAgent/Fast/Slow 模块）记录状态与错误。

返回约定：
- (data, status, exc)
    - `data`: 成功时为解析后的 JSON 字典，否则为 None；
    - `status`: 最近一次 HTTP 状态码（若无响应则可能为 None）；
    - `exc`: 最终失败时的异常对象（成功时为 None）。
"""

from __future__ import annotations

import random
import time
from typing import Any

import requests


_TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _response_excerpt(resp: Any, limit: int = 800) -> str:
    """提取响应体摘要，避免日志/异常信息过长。

    参数：
    - resp: requests.Response 或兼容对象；
    - limit: 最大截断长度。

    返回：
    - 去首尾空白后的文本摘要；若不可用则返回空字符串。
    """
    try:
        text = (getattr(resp, "text", "") or "").strip()
    except Exception:
        text = ""
    return text[:limit] if text else ""


def post_json_with_retry(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float | int | None,
    max_retries: int,
    session_env: requests.Session | None = None,
) -> tuple[dict[str, Any] | None, int | None, Exception | None]:
    """发送 JSON POST 请求，并提供最小重试能力。

    行为说明：
    1) 使用 `session_env`（若为空则内部创建一个 Session）；
    2) 对网络异常与瞬时 HTTP 状态码按 `max_retries` 重试；
    3) 成功时返回 JSON 字典；失败时返回最后一次状态码与异常。

    参数说明：
    - url: 请求地址；
    - headers: HTTP 头；
    - payload: JSON 请求体；
    - timeout: 单次请求超时；
    - max_retries: 最大尝试次数（<=0 时按 1 处理）；
    - session_env: 可选会话（用于复用连接）。

    返回：
    - (json_dict_or_none, status_code_or_none, exception_or_none)
    """

    # 统一保证至少尝试 1 次，避免 0/None 导致不发请求。
    attempts = max(1, int(max_retries or 1))

    # 标记是否由本函数创建 Session。
    # 如果是本函数创建，则在 finally 中关闭；外部传入的不关闭。
    created_env = False

    if session_env is None:
        session_env = requests.Session()
        created_env = True

    try:
        last_exc: Exception | None = None
        last_status: int | None = None

        for attempt in range(1, attempts + 1):
            sess = session_env

            try:
                resp = sess.post(url, headers=headers, json=payload, timeout=timeout)
                status = getattr(resp, "status_code", None)
                last_status = int(status) if status is not None else None
            except requests.exceptions.RequestException as e:
                # requests 网络层错误：按重试策略处理。
                last_exc = e
                if attempt < attempts:
                    _sleep_backoff(attempt)
                    continue
                break
            except Exception as e:
                # 非 requests 异常通常代表不可恢复错误，直接返回。
                return None, last_status, e

            if not getattr(resp, "ok", False):
                # 非 2xx：构造可诊断异常。
                body = _response_excerpt(resp)
                last_exc = RuntimeError(f"HTTP {last_status}: {body}") if body else RuntimeError(f"HTTP {last_status}")

                # 对瞬时状态码执行重试（如 429/5xx）。
                if last_status in _TRANSIENT_HTTP_STATUSES and attempt < attempts:
                    _sleep_backoff(attempt)
                    continue
                return None, last_status, last_exc

            try:
                data = resp.json()
            except Exception as e:
                # 响应体不是合法 JSON：记录摘要并按策略重试。
                body = _response_excerpt(resp)
                last_exc = RuntimeError(f"JSON decode failed: {e}; body: {body}") if body else e
                if attempt < attempts:
                    _sleep_backoff(attempt)
                    continue
                return None, last_status, last_exc

            # 成功路径：返回 JSON 与状态码。
            return data, last_status, None

        # 全部重试失败。
        return None, last_status, last_exc

    finally:
        # 仅关闭本函数内部创建的 Session，避免误关外部复用连接。
        if created_env:
            try:
                session_env.close()
            except Exception:
                pass


def _sleep_backoff(attempt: int) -> None:
    """指数退避 + 轻微随机抖动，减少并发重试冲突。"""
    sleep_s = min(8.0, (0.6 * (2 ** (max(0, attempt - 1)))) + random.random() * 0.2)
    time.sleep(sleep_s)
