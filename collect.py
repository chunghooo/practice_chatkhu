"""EVIDA — 유튜브 댓글 수집 (YouTube Data API v3)

video_id(url)                         → 11자리 영상 ID 또는 None
fetch(video_id, key, limit, progress)  → (rows, raw_pages)
  - rows: 공통 형식 dict 목록 (댓글 ID 기준 중복 제거, limit에서 자름)
  - raw_pages: API 응답 원본을 그대로 보관 (증거 무결성용)
  - 수집 도중 오류가 나면: 이미 받은 댓글이 있으면 그것을 돌려주고
    raw_pages 마지막에 {"error": 안내문}을 남긴다. 받은 게 없으면 CollectError.
"""
import re
from datetime import datetime, timezone

import requests

API = "https://www.googleapis.com/youtube/v3"
TIMEOUT = 20

ERRORS = {
    "commentsDisabled": "이 영상은 댓글이 꺼져 있어요.",
    "videoNotFound": "영상을 찾을 수 없어요.",
    "quotaExceeded": "오늘 API 사용량을 다 썼어요.",
    "keyInvalid": "API 키가 올바르지 않아요.",
}


class CollectError(Exception):
    """사용자에게 그대로 보여줄 한국어 오류."""


# ---------------------------------------------------------------- 영상 ID
_ID = r"([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])"
_PATTERNS = [
    re.compile(r"[?&]v=" + _ID),
    re.compile(r"youtu\.be/" + _ID),
    re.compile(r"/shorts/" + _ID),
    re.compile(r"/embed/" + _ID),
    re.compile(r"/live/" + _ID),
]


def video_id(url):
    """watch?v=, youtu.be/, shorts/, embed/, live/ 형식에서 11자리 ID 추출. 실패하면 None."""
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):  # ID만 붙여넣은 경우
        return url
    for p in _PATTERNS:
        m = p.search(url)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------- API 호출
def _error_message(status, data):
    err = data.get("error", {}) if isinstance(data, dict) else {}
    if not isinstance(err, dict):
        err = {}
    reasons = [e.get("reason", "") for e in err.get("errors", []) if isinstance(e, dict)]
    blob = str(err)
    for k, msg in ERRORS.items():
        if k in reasons:
            return msg
    if "API_KEY_INVALID" in blob or "API key not valid" in blob or "keyExpired" in reasons:
        return ERRORS["keyInvalid"]
    if "dailyLimitExceeded" in reasons or "rateLimitExceeded" in reasons:
        return ERRORS["quotaExceeded"]
    if status == 404:
        return ERRORS["videoNotFound"]
    return f"유튜브 API 오류가 발생했어요. ({status}: {err.get('message') or '알 수 없는 오류'})"


def _get(endpoint, params, key):
    try:
        r = requests.get(f"{API}/{endpoint}", params={**params, "key": key}, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise CollectError(f"네트워크 오류로 유튜브에 연결하지 못했어요. ({e.__class__.__name__})")
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code != 200 or (isinstance(data, dict) and "error" in data):
        raise CollectError(_error_message(r.status_code, data))
    return data


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- 형식 변환
def _row(c, vid):
    s = c.get("snippet", {}) or {}
    cid = c.get("id", "")
    text = s.get("textOriginal")
    if text is None:
        text = s.get("textDisplay", "")
    return {
        "id": cid,
        "author": s.get("authorDisplayName", ""),
        "channel": s.get("authorChannelUrl", ""),
        "cid": (s.get("authorChannelId") or {}).get("value", ""),
        "text": text,  # 원문 그대로 (수정·요약 금지)
        "time": s.get("publishedAt", ""),
        "link": f"https://www.youtube.com/watch?v={vid}&lc={cid}",
        "label": "",
        "severity": 0,
        "category": "",
        "reason": "",
        "law": "",
        "statute": "",
        "ai": None,
        "gem": "",
        "memo": "",
    }


# ---------------------------------------------------------------- 수집
def fetch(video_id, key, limit=3000, progress=None):
    """댓글·답글을 limit개까지 수집한다. 반환: (rows, raw_pages)"""
    if not key:
        raise CollectError(ERRORS["keyInvalid"])
    if not video_id:
        raise CollectError(ERRORS["videoNotFound"])
    limit = max(1, int(limit))

    rows, raw_pages, seen = [], [], set()

    def full():
        return len(rows) >= limit

    def add(c):
        if full():
            return
        cid = c.get("id")
        if not cid or cid in seen:
            return
        seen.add(cid)
        rows.append(_row(c, video_id))

    def tick():
        if progress:
            try:
                progress(len(rows))
            except Exception:
                pass

    def call(endpoint, params):
        data = _get(endpoint, params, key)
        raw_pages.append({
            "endpoint": endpoint,
            "request": dict(params),  # API 키는 저장하지 않음
            "fetched_at": _now(),
            "response": data,  # 응답 원본 그대로
        })
        return data

    def all_replies(parent_id):
        out, token = [], None
        while True:
            params = {"part": "snippet", "parentId": parent_id,
                      "maxResults": 100, "textFormat": "plainText"}
            if token:
                params["pageToken"] = token
            data = call("comments", params)
            out.extend(data.get("items", []))
            token = data.get("nextPageToken")
            if not token or len(rows) + len(out) >= limit:
                return out

    token = None
    try:
        while not full():
            params = {"part": "snippet,replies", "videoId": video_id,
                      "maxResults": 100, "textFormat": "plainText"}
            if token:
                params["pageToken"] = token
            data = call("commentThreads", params)

            for item in data.get("items", []):
                if full():
                    break
                sn = item.get("snippet", {}) or {}
                top = sn.get("topLevelComment") or {}
                add(top)

                replies = (item.get("replies") or {}).get("comments", []) or []
                total = int(sn.get("totalReplyCount", 0) or 0)
                if total > len(replies) and top.get("id") and not full():
                    replies = all_replies(top["id"])
                for rp in replies:
                    add(rp)
                tick()

            tick()
            token = data.get("nextPageToken")
            if not token:
                break
    except CollectError as e:
        if not rows:
            raise
        raw_pages.append({"error": str(e), "fetched_at": _now()})

    rows = rows[:limit]
    tick()
    return rows, raw_pages


# ---------------------------------------------------------------- 자체 점검
if __name__ == "__main__":
    import os
    import sys

    cases = {
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=10": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=abc": "dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ?feature=shared": "dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=short": None,
        "https://example.com": None,
        "": None,
    }
    bad = 0
    for u, want in cases.items():
        got = video_id(u)
        ok = got == want
        bad += not ok
        print(("OK  " if ok else "FAIL"), repr(u), "→", got)
    print(f"video_id: {len(cases) - bad}/{len(cases)} 통과")

    key = os.environ.get("YOUTUBE_API_KEY")
    if key and len(sys.argv) > 1:
        vid = video_id(sys.argv[1])
        try:
            rows, raw = fetch(vid, key, 300, progress=lambda n: print(f"\r{n}개", end=""))
            print(f"\n수집 {len(rows)}개, 원본 페이지 {len(raw)}개")
            for r in rows[:3]:
                print(r["time"], r["author"], r["text"][:40].replace("\n", " "))
        except CollectError as e:
            print("오류:", e)
    sys.exit(1 if bad else 0)
