"""EVIDA 화면 (설계서 [4단계] 기준) — collect.py, detect.py, report.py를 불러서 연결한다."""
import hashlib, io, json, re
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import streamlit as st

from collect import video_id, fetch
from detect import classify, train_model, apply_model
from report import make_pdf

KST = timezone(timedelta(hours=9))
GEM = "https://generativelanguage.googleapis.com/v1beta/"
GEM_PROMPT = """너는 한국어 댓글이 특정인을 향한 악플인지 판단하는 검토자야.
악플: 특정인·집단을 겨냥한 모욕, 비하, 조롱, 비꼬기, 외모 비하, 위협, 성희롱, 2차 가해, 근거 없는 의혹.
아님: 칭찬·감탄 속 욕, 혼잣말 감탄, 게임 플레이 묘사, 인용, 예의 있는 비판. 욕이 없어도 특정인을 깎아내리면 악플.
JSON 배열로만 답해: [{"i": 번호, "v": "악플" 또는 "아님", "r": "15자 이내 이유"}]
댓글:
"""


def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return ""


@st.cache_resource(show_spinner="AI 모델 학습 중… (처음 한 번만)")
def get_model():
    try:
        return train_model()
    except Exception:
        return None


def dt(t):
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).astimezone(KST).strftime("%Y.%m.%d %H:%M")
    except Exception:
        return ""


def ocr(b):
    try:
        import pytesseract
        from PIL import Image, ImageOps, ImageStat
        im = ImageOps.grayscale(Image.open(io.BytesIO(b)))
        if ImageStat.Stat(im).mean[0] < 128:
            im = ImageOps.invert(im)
        k = max(1, min(3, 1400 / im.width))
        im = im.resize((int(im.width * k), int(im.height * k)))
        t = pytesseract.image_to_string(im, lang="kor+eng", config="--psm 6")
    except Exception:
        return ""
    return "\n".join(l.strip() for l in t.splitlines() if len(re.findall(r"[가-힣]", l)) >= 2)


def analyze(rows, raw, src, url="", imgs=()):
    for r in rows:
        r.update(classify(r["text"]))
        r.setdefault("ai", None), r.setdefault("gem", ""), r.setdefault("memo", "")
    m = get_model()
    if m is not None:
        out = apply_model(rows, m)
        rows = out if isinstance(out, list) else rows
    st.session_state.update(
        rows=rows, raw=raw, src=src, url=url, imgs=list(imgs), gcache={}, ver=st.session_state.get("ver", 0) + 1,
        sha=hashlib.sha256(raw + b"".join(imgs)).hexdigest(), at=datetime.now(timezone.utc).isoformat(),
        sel={r["id"] for r in rows if r.get("label") == "악플" and r.get("severity", 0) >= 2})


def gem_review(texts, key):
    cache = st.session_state.gcache
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if not todo:
        return cache
    ms = requests.get(GEM + "models", params={"key": key}, timeout=20).json().get("models", [])
    names = [m["name"] for m in ms if "generateContent" in m.get("supportedGenerationMethods", [])]
    model = sorted([n for n in names if "flash-lite" in n] or [n for n in names if "flash" in n] or names)[-1]
    for i in range(0, len(todo), 50):
        part = todo[i:i + 50]
        body = "\n".join(f"{j}. {t[:300]}" for j, t in enumerate(part))
        r = requests.post(f"{GEM}{model}:generateContent", params={"key": key}, timeout=60, json={
            "contents": [{"parts": [{"text": GEM_PROMPT + body}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        if r.status_code == 429:
            raise RuntimeError("오늘 Gemini 무료 사용량을 다 썼어요.")
        r.raise_for_status()
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        for x in json.loads(txt[txt.find("["):txt.rfind("]") + 1]):
            if 0 <= int(x["i"]) < len(part):
                cache[part[int(x["i"])]] = (x.get("v") == "악플", x.get("r", ""))
    return cache


# ---------- 화면 ----------
st.set_page_config("EVIDA", "🛡️", layout="wide")
st.markdown("<style>[data-testid='stToolbar']{display:none}.block-container{max-width:1150px}</style>",
            unsafe_allow_html=True)
st.title("🛡️ EVIDA")
st.caption("AI 악플 분석 및 증거자료 생성 서비스 — 신고·고소 준비용 참고자료를 만듭니다.")
for col, (h, b) in zip(st.columns(3), [("① 넣기", "유튜브 주소 또는 스크린샷·텍스트"), ("② 자동 분류", "규칙 + 학습 AI (+ 생성형 AI)"),
                                       ("③ 보고서 받기", "선택 → 사건 정보 → PDF")]):
    col.container(border=True).markdown(f"**{h}**  \n{b}")

t1, t2 = st.tabs(["🔗 유튜브 링크", "📋 텍스트·스크린샷"])
with t1:
    key = secret("YOUTUBE_API_KEY") or st.text_input("YouTube API 키", type="password")
    a, b, c = st.columns([5, 3, 1.3], vertical_alignment="bottom")
    url = a.text_input("유튜브 영상 주소", placeholder="https://youtu.be/...")
    limit = b.number_input("최대 수집 댓글 수", 100, 50000, 3000, 1000)
    if c.button("수집 시작", type="primary"):
        vid = video_id(url)
        if not (vid and key):
            st.error("주소와 API 키를 확인하세요.")
        else:
            bar = st.progress(0.0, "댓글 수집 중…")
            try:
                rows, raw = fetch(vid, key, limit, lambda n: bar.progress(min(n / limit, 1.0), f"댓글 {n:,}개 수집 중…"))
                bar.progress(1.0, "분류 중…")
                analyze(rows, json.dumps(raw, ensure_ascii=False).encode(), "유튜브", url)
                bar.empty()
            except Exception as e:
                st.error(f"수집 실패: {e}")
with t2:
    a, b = st.columns(2)
    imgs = a.file_uploader("스크린샷 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if imgs and a.button("사진에서 글자 읽기"):
        st.session_state.txt = "\n".join(ocr(f.getvalue()) for f in imgs)
    txt = b.text_area("댓글·메시지 (한 줄에 하나씩)", key="txt", height=200)
    if b.button("분류하기", type="primary"):
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if lines:
            analyze([dict(id=f"m{i}", author="", channel="", cid="", text=l, time="", link="") for i, l in enumerate(lines, 1)],
                    txt.encode(), "직접 입력", imgs=[f.getvalue() for f in imgs or []])
        else:
            st.error("내용을 입력하세요.")

if "rows" in st.session_state:
    S = st.session_state
    df = pd.DataFrame(S.rows)
    for col in ["severity", "ai", "gem", "memo", "category", "law", "reason", "author", "time", "link"]:
        if col not in df:
            df[col] = None
    df["when"] = pd.to_datetime(df.time, errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
    bad = df.label == "악플"
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("출처", S.src)
    m2.metric("전체", f"{len(df):,}")
    m3.metric("악플 후보", f"{bad.sum():,}")
    m4.metric("AI가 찾은 것", f"{(df.category == 'AI 판단').sum():,}")

    with st.expander("📊 악플 현황", expanded=True):
        if bad.any():
            g1, g2 = st.columns(2)
            g1.markdown("**유형별 악플 수**")
            g1.bar_chart(df[bad].category.fillna("").str.split(", ").explode().value_counts(), color="#FF4B4B")
            g2.markdown("**상습 작성자 Top 5**")
            g2.dataframe(df[bad & df.author.fillna("").ne("")].author.value_counts().head(5), use_container_width=True)
            if df.when.notna().any():
                st.markdown("**날짜별 댓글 수 (전체 vs 악플)**")
                st.bar_chart(pd.DataFrame({"전체": df.when.dt.date.value_counts(),
                                           "악플": df[bad].when.dt.date.value_counts()}).fillna(0).sort_index(), stack=False)
        else:
            st.caption("악플 후보가 없어요.")
        st.caption("학습 데이터 출처: Smilegate AI UnSmile(CC BY-NC-ND 4.0), K-MHaS(CC BY-SA 4.0)")

    with st.expander("🤖 생성형 AI 2차 검토 (Gemini)", expanded=False):
        gkey = secret("GEMINI_API_KEY")
        mode = st.radio("검토 범위", ["애매한 것만 (절약)", "악플 후보 전부 (정밀)"], horizontal=True)
        ai_ = pd.to_numeric(df.ai, errors="coerce").fillna(0)
        doubt = (bad & (df.severity == 1)) | (~bad & ai_.between(40, 64))
        tgt = df[doubt] if mode.startswith("애매") else df[bad | doubt]
        texts = list(dict.fromkeys(tgt.text))[:200]
        st.caption(f"보낼 댓글 {len(texts)}개 (같은 문장은 1번만, 50개씩 묶어 요청)")
        if not gkey:
            st.caption("GEMINI_API_KEY가 없어 사용할 수 없어요.")
        if st.button("재검토 실행", disabled=not (gkey and texts)):
            try:
                with st.spinner("Gemini가 검토 중…"):
                    res = gem_review(texts, gkey)
                for r in S.rows:
                    if r["text"] in res and r["text"] in texts:
                        is_bad, why = res[r["text"]]
                        r["gem"] = ("악플: " if is_bad else "아님: ") + why
                        if is_bad:
                            r["label"], r["severity"] = "악플", max(r.get("severity") or 0, 2)
                            S.sel.add(r["id"])
                        elif r.get("label") == "악플":
                            r["severity"] = 1
                            S.sel.discard(r["id"])
                S.ver += 1
                st.rerun()
            except Exception as e:
                st.error(f"재검토 실패: {e}")

    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.6, 1.4])
    scope = f1.radio("범위", ["악플만", "전체", "선택한 것만"], horizontal=True)
    cats = f2.multiselect("유형", sorted({c.strip() for x in df.category.dropna() for c in str(x).split(",") if c.strip()}))
    q = f3.text_input("검색어 (쉼표로 여러 개)")
    days = df.when.dropna().dt.date
    rng = f4.date_input("작성 날짜", (days.min(), days.max())) if len(days) else ()
    v = {"전체": df, "악플만": df[bad], "선택한 것만": df[df.id.isin(S.sel)]}[scope]
    if cats:
        v = v[v.category.fillna("").apply(lambda c: any(x in c for x in cats)).astype(bool)]
    for w in [w.strip() for w in q.split(",") if w.strip()]:
        v = v[v.text.str.contains(re.escape(w), na=False)]
    if len(rng) == 2:
        v = v[v.when.dt.date.between(*rng) | v.when.isna()]
    v = v.sort_values("severity", ascending=False).copy()
    v.insert(0, "포함", v.id.isin(S.sel))
    v["when"] = v.when.dt.strftime("%Y-%m-%d %H:%M").fillna("")
    st.caption(f"{len(v):,}개 표시 · ✅ 선택 {len(S.sel)}건 · 심각도 1 = 애매, 2 = 명백, 3 = 심각")
    cols = ["포함", "severity", "law", "author", "when", "text", "memo", "reason", "ai", "gem", "link"]
    ed = st.data_editor(v[cols], hide_index=True, key=f"ed{S.ver}{scope}{cats}{q}{rng}",
                        disabled=[c for c in cols if c not in ("포함", "memo")],
                        column_config={"severity": "심각도", "law": "법적 검토", "author": "작성자", "when": "작성 시각(KST)",
                                       "text": st.column_config.TextColumn("내용", width="large"), "memo": "메모",
                                       "reason": "판단 근거", "ai": "AI 점수", "gem": "생성형 AI 의견",
                                       "link": st.column_config.LinkColumn("원문", display_text="열기")})
    memo = {r["id"]: r.get("memo") or "" for r in S.rows}
    for i, r in ed.iterrows():
        cid = v.at[i, "id"]
        (S.sel.add if r["포함"] else S.sel.discard)(cid)
        memo[cid] = r["memo"] or ""
    for r in S.rows:
        r["memo"] = memo.get(r["id"], "")
    sel = sorted([r for r in S.rows if r["id"] in S.sel], key=lambda r: r.get("time") or "")

    # ---------- 증거자료 생성 ----------
    st.divider()
    h1, h2 = st.columns([3, 2], vertical_alignment="center")
    h1.subheader("📄 증거자료 생성")
    h2.info("이 보고서는 경찰청 공식 서식이 아닌, 신고 및 고소 준비를 위한 참고자료입니다.")
    sd = sorted(d for d in (pd.to_datetime(r.get("time"), errors="coerce", utc=True) for r in sel) if pd.notna(d))
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**① 기본 정보 입력**")
        title = st.text_input("사건명 *", placeholder="예) 유튜브 게시물 내 악성 댓글 관련 사건")
        plat = st.selectbox("플랫폼 *", ["YouTube", "Instagram", "KakaoTalk", "기타"], index=0 if S.src == "유튜브" else 3)
        st.text_input("분석된 악성 댓글 수", f"{len(sel)}건 (자동 입력)", disabled=True)
        per = st.date_input("피해 발생 기간", (sd[0].date(), sd[-1].date()) if sd else (), key=f"per{S.ver}")
    with c2.container(border=True):
        st.markdown("**② 사건 내용 작성**")
        story = st.text_area("피해 발생 경위 *", height=110)
        harm = st.text_area("피해 내용 *", height=110)
        note = st.text_area("추가 설명 (선택)", height=70)
    case = dict(title=title or "온라인 댓글 관련 증거 정리", platform=plat, story=story, harm=harm, note=note,
                period=" ~ ".join(x.strftime("%Y.%m.%d") for x in per) if len(per) == 2 else "")
    with st.container(border=True):
        a, b = st.columns([3, 1], vertical_alignment="center")
        a.markdown("**③ 보고서 미리보기**")
        if b.toggle("미리보기 열기"):
            st.dataframe(pd.DataFrame([dict(번호=f"{i:03d}", 작성자=r.get("author") or "미확인", 작성시각=dt(r.get("time")) or "미확인",
                                            댓글원문=r["text"], URL=r.get("link") or "미확인") for i, r in enumerate(sel, 1)]),
                         hide_index=True)
    miss = [n for n, x in [("증거 선택", sel), ("피해 발생 경위", story.strip()), ("피해 내용", harm.strip())] if not x]
    with st.container(border=True):
        a, b = st.columns([3, 1], vertical_alignment="center")
        a.markdown("**④ PDF 다운로드**")
        if miss:
            a.caption("필수 항목을 채워주세요: " + ", ".join(miss))
            b.button("⬇️ 증거 정리 보고서 다운로드", disabled=True)
        else:
            meta = dict(src=S.src, url=S.url, collected_at=S.at, sha256=S.sha, total=len(df), images=S.imgs)
            try:
                b.download_button("⬇️ 증거 정리 보고서 다운로드", make_pdf(sel, meta, case), "EVIDA_증거정리보고서.pdf",
                                  "application/pdf", type="primary")
            except Exception as e:
                st.error(f"PDF 생성 실패 (report.py 확인 필요): {e}")
    d1, d2 = st.columns(2)
    d1.download_button("📊 전체 목록(CSV)", df.drop(columns=["when"]).to_csv(index=False).encode("utf-8-sig"), "EVIDA_목록.csv")
    d2.download_button("🔒 원본 데이터", S.raw, "EVIDA_원본." + ("json" if S.src == "유튜브" else "txt"))
    st.caption(f"원본 SHA-256: {S.sha}")
