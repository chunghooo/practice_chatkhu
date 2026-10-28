"""EVIDA — PDF 증거 정리 보고서 (ReportLab, A4)

make_pdf(selected, meta, case) -> bytes
  selected : 공통 형식 댓글 dict 목록 (보고서에 넣을 증거)
  meta     : src, url, collected_at, sha256, total, images
             images = 캡처 이미지(bytes). 다음 중 어느 형식이든 된다.
               - {댓글ID: bytes 또는 [bytes, ...]}
               - [bytes, ...]  (selected 순서와 같은 위치끼리 연결)
             댓글 dict 안에 "image"(bytes) 키가 있어도 사용한다.
  case     : title, platform, period, story, harm, note

원칙: 원문(text)은 수정·요약하지 않는다. 없는 값은 "미확인", 빈 서술은 "미기재".
      법적 판단은 단정하지 않는다("의심/검토/참고").
"""
import io
import os
import re
from datetime import date, datetime, timedelta, timezone
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# =====================================================================
# 색·상수
# =====================================================================
C_HEAD = colors.HexColor("#F3F4F6")     # 표 머리·왼쪽 칸
C_LINE = colors.HexColor("#D1D5DB")     # 선
C_BAND = colors.HexColor("#E8EAFD")     # 증거번호 띠·AI 상자
C_LINK = colors.HexColor("#3B3FD8")     # 링크
C_BOX = colors.HexColor("#F9FAFB")      # 상자 배경
C_GREY = colors.HexColor("#6B7280")
C_TEXT = colors.HexColor("#111827")
C_AVATAR = colors.HexColor("#C7CBF5")

KST = timezone(timedelta(hours=9))
PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X
NA = "미확인"
EMPTY = "미기재"
DISCLAIMER = "(본 내용은 참고용이며, 실제 법적 판단은 수사기관의 결정에 따릅니다.)"
CARD_TEXT_MAX = 700  # 재구성 카드에만 적용. 원문 전체는 위 표에 그대로 실린다.

# =====================================================================
# 폰트
# =====================================================================
FONT_DIR = "/usr/share/fonts/truetype/nanum"
REG_PATHS = [os.path.join(FONT_DIR, "NanumGothic.ttf"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "NanumGothic.ttf")]
BOLD_PATHS = [os.path.join(FONT_DIR, "NanumGothicBold.ttf"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "NanumGothicBold.ttf")]

_FONT = {"reg": None, "bold": None, "cmap": None}


def _register_fonts():
    if _FONT["reg"]:
        return
    reg = next((p for p in REG_PATHS if os.path.exists(p)), None)
    bold = next((p for p in BOLD_PATHS if os.path.exists(p)), None)
    if reg:
        f = TTFont("EvidaGothic", reg)
        pdfmetrics.registerFont(f)
        _FONT["reg"] = "EvidaGothic"
        _FONT["cmap"] = set(getattr(f.face, "charToGlyph", {}).keys()) or None
        if bold:
            pdfmetrics.registerFont(TTFont("EvidaGothicBold", bold))
            _FONT["bold"] = "EvidaGothicBold"
        else:
            _FONT["bold"] = "EvidaGothic"
    else:  # 나눔 폰트가 없을 때의 비상용 (packages.txt에 fonts-nanum 필요)
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
        _FONT["reg"] = _FONT["bold"] = "HYGothic-Medium"
        _FONT["cmap"] = None


# =====================================================================
# 글자 처리
# =====================================================================
_INVISIBLE = re.compile("[\u200b-\u200f\u2060\ufe00-\ufe0f\U000e0000-\U000e007f]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\U0001F900-\U0001F9FF]")


def safe(s):
    """폰트에 없는 글자(이모지 등)를 □로 바꾼다. 보이지 않는 결합 문자는 지운다."""
    s = _CTRL.sub("", _INVISIBLE.sub("", str(s)))
    cmap = _FONT["cmap"]
    if cmap:
        return "".join(ch if (ch in "\n\t " or ord(ch) in cmap) else "□" for ch in s)
    return _EMOJI.sub("□", "".join(ch if ord(ch) <= 0xFFFF else "□" for ch in s))


def para_text(s, empty=NA):
    """Paragraph용 마크업: 이스케이프 + 줄바꿈 보존."""
    if s is None or (isinstance(s, str) and not s.strip()):
        s = empty
    s = escape(safe(s)).replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\n", "<br/>")


def val(v, empty=NA):
    return empty if v is None or (isinstance(v, str) and not v.strip()) else v


# =====================================================================
# 시각
# =====================================================================
def _parse_dt(v):
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, date):
        dt = datetime(v.year, v.month, v.day)
    else:
        s = str(v).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def kst(v, fmt="%Y.%m.%d %H:%M"):
    dt = _parse_dt(v)
    return dt.astimezone(KST).strftime(fmt) if dt else NA


def utc(v, fmt="%Y.%m.%d %H:%M:%S"):
    dt = _parse_dt(v)
    return dt.astimezone(timezone.utc).strftime(fmt) if dt else NA


def period_text(p):
    if p is None or p == "" or p == []:
        return NA
    if isinstance(p, (list, tuple)):
        ds = [x for x in p if x]
        if not ds:
            return NA

        def d(x):
            if isinstance(x, (date, datetime)):
                return x.strftime("%Y.%m.%d")
            dt = _parse_dt(x)
            return dt.strftime("%Y.%m.%d") if dt else str(x)
        a, b = d(ds[0]), d(ds[-1])
        return a if a == b else f"{a} ~ {b}"
    if isinstance(p, (date, datetime)):
        return p.strftime("%Y.%m.%d")
    return str(p)


# =====================================================================
# 스타일
# =====================================================================
def _styles():
    R, B = _FONT["reg"], _FONT["bold"]
    base = dict(fontName=R, textColor=C_TEXT, wordWrap="CJK")
    return {
        "h2": ParagraphStyle("h2", fontName=B, fontSize=12, leading=16, textColor=C_TEXT,
                             spaceBefore=8, spaceAfter=6, wordWrap="CJK"),
        "body": ParagraphStyle("body", fontSize=9.5, leading=14, **base),
        "cell": ParagraphStyle("cell", fontSize=9, leading=13, **base),
        "cellb": ParagraphStyle("cellb", fontName=B, fontSize=9, leading=13, textColor=C_TEXT,
                                wordWrap="CJK"),
        "small": ParagraphStyle("small", fontSize=8.5, leading=12.5, textColor=C_GREY,
                                fontName=R, wordWrap="CJK"),
        "tiny": ParagraphStyle("tiny", fontSize=7.8, leading=11.5, textColor=C_GREY,
                               fontName=R, wordWrap="CJK"),
        "th": ParagraphStyle("th", fontName=B, fontSize=8.8, leading=12, textColor=C_TEXT,
                             alignment=TA_CENTER, wordWrap="CJK"),
        "td": ParagraphStyle("td", fontSize=8.3, leading=11.8, **base),
        "tdc": ParagraphStyle("tdc", fontSize=8.3, leading=11.8, alignment=TA_CENTER, **base),
        "link": ParagraphStyle("link", fontSize=7.8, leading=11, textColor=C_LINK, fontName=R,
                               wordWrap="CJK"),
        "band": ParagraphStyle("band", fontName=B, fontSize=11, leading=15, textColor=C_TEXT,
                               alignment=TA_LEFT, wordWrap="CJK"),
        "aih": ParagraphStyle("aih", fontName=B, fontSize=9.5, leading=14, textColor=C_TEXT,
                              wordWrap="CJK"),
        "card_name": ParagraphStyle("card_name", fontName=B, fontSize=9, leading=13,
                                    textColor=C_TEXT, wordWrap="CJK"),
        "card_text": ParagraphStyle("card_text", fontSize=9.5, leading=14, **base),
        "card_act": ParagraphStyle("card_act", fontSize=8, leading=11, textColor=C_GREY,
                                   fontName=R, wordWrap="CJK"),
    }


def link(url, st):
    if not url or not str(url).strip():
        return Paragraph(NA, st["td"])
    u = str(url).strip()
    href = escape(u, {'"': "&quot;"})
    # 긴 URL 줄바꿈을 돕기 위해 ? & = / 뒤에 폭 0 공백 대신 줄바꿈 가능 지점을 둔다(CJK 줄바꿈)
    return Paragraph(f'<a href="{href}" color="#3B3FD8">{escape(safe(u))}</a>', st["link"])


# =====================================================================
# 블록
# =====================================================================
def kv_table(rows, st, label_w=38 * mm):
    data = [[Paragraph(para_text(k), st["cellb"]), v if not isinstance(v, str)
             else Paragraph(para_text(v), st["cell"])] for k, v in rows]
    t = Table(data, colWidths=[label_w, CONTENT_W - label_w])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), C_HEAD),
        ("GRID", (0, 0), (-1, -1), 0.6, C_LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def box(flowables, bg=C_BOX, border=C_LINE, pad=10):
    t = Table([[flowables]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.6, border),
        ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LEFTPADDING", (0, 0), (-1, -1), pad + 2), ("RIGHTPADDING", (0, 0), (-1, -1), pad + 2),
    ]))
    return t


def text_box(text, st):
    """긴 서술도 쪽을 넘길 수 있도록 문단 단위로 나눈 회색 상자."""
    s = "" if text is None else str(text)
    if not s.strip():
        return [box([Paragraph(EMPTY, st["body"])])]
    parts = [p for p in re.split(r"\n\s*\n", s.replace("\r\n", "\n"))]
    out, chunk = [], []
    for p in parts:
        chunk.append(Paragraph(para_text(p, ""), st["body"]))
        chunk.append(Spacer(1, 4))
        if len(chunk) >= 8:
            out.append(box(chunk[:-1]))
            chunk = []
    if chunk:
        out.append(box(chunk[:-1]))
    return out


def band(text, st):
    t = Table([[Paragraph(para_text(text), st["band"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BAND),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def avatar(name, size=10 * mm):
    d = Drawing(size, size)
    d.add(Circle(size / 2, size / 2, size / 2, fillColor=C_AVATAR, strokeColor=None))
    initial = safe((str(name or "?").lstrip("@").strip() or "?")[0])
    d.add(String(size / 2, size / 2 - 3.5, initial, fontName=_FONT["bold"], fontSize=10,
                 fillColor=colors.white, textAnchor="middle"))
    return d


def comment_card(r, st):
    """실제 캡처가 아닌 '댓글 화면 재구성' 카드."""
    text = "" if r.get("text") is None else str(r.get("text"))
    shown = text
    tail = ""
    if len(text) > CARD_TEXT_MAX:
        shown = text[:CARD_TEXT_MAX]
        tail = "<br/><font color='#6B7280'>… (이하 생략 — 원문 전체는 위 표의 '댓글 원문' 참조)</font>"
    name = val(r.get("author"))
    right = [
        Paragraph(f"{para_text(name)} &nbsp;<font color='#6B7280' size='8'>"
                  f"{kst(r.get('time'))}</font>", st["card_name"]),
        Spacer(1, 3),
        Paragraph(para_text(shown, NA) + tail, st["card_text"]),
        Spacer(1, 5),
        Paragraph("좋아요 · 싫어요 · 답글", st["card_act"]),
    ]
    t = Table([[avatar(name), right]], colWidths=[14 * mm, CONTENT_W - 14 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, C_LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def _to_bytes(x):
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if hasattr(x, "getvalue"):  # Streamlit UploadedFile, BytesIO
        return x.getvalue()
    if hasattr(x, "read"):
        return x.read()
    if isinstance(x, str) and os.path.exists(x):
        with open(x, "rb") as f:
            return f.read()
    return None


def images_for(r, i, meta):
    imgs = meta.get("images") if isinstance(meta, dict) else None
    found = []
    if r.get("image") is not None:
        found.append(r["image"])
    if isinstance(imgs, dict):
        v = imgs.get(r.get("id"))
        if v is None:
            v = imgs.get(i, imgs.get(str(i)))
        if v is not None:
            found.extend(v if isinstance(v, (list, tuple)) else [v])
    elif isinstance(imgs, (list, tuple)) and i < len(imgs) and imgs[i] is not None:
        v = imgs[i]
        found.extend(v if isinstance(v, (list, tuple)) else [v])
    return [b for b in (_to_bytes(x) for x in found) if b]


def image_flowable(data, max_w=CONTENT_W - 24, max_h=105 * mm):
    try:
        from PIL import Image as PILImage
        im = PILImage.open(io.BytesIO(data))
        im.load()
        if im.mode not in ("RGB", "L"):
            bg = PILImage.new("RGB", im.size, "white")
            bg.paste(im, mask=im.convert("RGBA").split()[-1])
            im = bg
        w, h = im.size
        scale = min(max_w / w, max_h / h, 1.0 if w > 400 else max_w / w)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return Image(buf, width=w * scale, height=h * scale)
    except Exception:
        return None


def ai_note(r, st):
    law = val(r.get("law"), "")
    statute = val(r.get("statute"), "")
    reason = val(r.get("reason"), "")
    lines = []
    if law:
        s = f"이 댓글은 <b>{para_text(law)}</b> 사항으로 검토해 볼 수 있습니다"
        s += f"(관련 조항: {para_text(statute)})." if statute else "."
        lines.append(s)
    else:
        lines.append("자동 분류에서 법적 검토 유형이 정해지지 않았습니다(미확인). "
                     "사용자가 직접 증거로 선택한 댓글입니다.")
    if reason:
        lines.append(f"판단 근거: {para_text(reason)}")
    extra = []
    if r.get("ai") is not None and str(r.get("ai")) != "":
        extra.append(f"학습형 AI 악플 가능성 {para_text(r.get('ai'))}점(100점 만점)")
    if r.get("gem"):
        extra.append(f"생성형 AI 의견: {para_text(r.get('gem'))}")
    if extra:
        lines.append(" / ".join(extra))
    if r.get("memo"):
        lines.append(f"사용자 메모: {para_text(r.get('memo'))}")
    body = [Paragraph("AI 분석 참고사항", st["aih"]), Spacer(1, 4)]
    body += [Paragraph(x, st["body"]) for x in lines]
    body += [Spacer(1, 4), Paragraph(DISCLAIMER, st["small"])]
    return box(body, bg=C_BAND, border=C_BAND)


# =====================================================================
# 머리말·꼬리말
# =====================================================================
def _on_page(c, doc):
    c.saveState()
    c.setFont(_FONT["bold"], 15)
    c.setFillColor(C_TEXT)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 15 * mm, "EVIDA 증거 정리 보고서")
    c.setFont(_FONT["reg"], 8.5)
    c.setFillColor(C_GREY)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 20.5 * mm, "신고 및 고소 준비 참고자료")
    c.setStrokeColor(C_LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN_X, PAGE_H - 24 * mm, PAGE_W - MARGIN_X, PAGE_H - 24 * mm)
    c.line(MARGIN_X, 15 * mm, PAGE_W - MARGIN_X, 15 * mm)
    c.setFont(_FONT["reg"], 8)
    c.drawString(MARGIN_X, 10 * mm, "EVIDA | AI 악플 분석 및 증거자료 생성 서비스")
    c.drawRightString(PAGE_W - MARGIN_X, 10 * mm, f"{doc.page}")
    c.restoreState()


# =====================================================================
# 본문
# =====================================================================
def _overview(selected, case, st):
    out = [Paragraph("1. 사건 개요", st["h2"]),
           kv_table([("사건명", val(case.get("title"))),
                     ("플랫폼", val(case.get("platform"))),
                     ("분석된 악성 댓글 수", f"{len(selected)}건"),
                     ("피해 발생 기간", period_text(case.get("period")))], st),
           Spacer(1, 8)]
    for title, key in [("2. 피해 발생 경위", "story"), ("3. 피해 내용", "harm"),
                       ("4. 추가 설명", "note")]:
        out += [Paragraph(title, st["h2"])] + text_box(case.get(key), st) + [Spacer(1, 6)]
    return out


def _evidence_table(selected, st):
    head = [Paragraph(h, st["th"]) for h in ["번호", "작성자", "작성 시각", "댓글 원문", "URL"]]
    widths = [12 * mm, 28 * mm, 26 * mm, 66 * mm, CONTENT_W - 132 * mm]
    data = [head]
    for i, r in enumerate(selected, 1):
        parts = _chunks(r.get("text"), 20, 30)
        data.append([
            Paragraph(f"{i:03d}", st["tdc"]),
            Paragraph(para_text(r.get("author")), st["td"]),
            Paragraph(kst(r.get("time")), st["tdc"]),
            Paragraph(para_text(_show(parts[0])), st["td"]),
            link(r.get("link"), st),
        ])
        for c in parts[1:]:  # 아주 긴 원문은 다음 행에 이어서 싣는다
            data.append([Paragraph(f"{i:03d}<br/><font size='6.5'>이어짐</font>", st["tdc"]), "", "",
                         Paragraph(para_text(_show(c), ""), st["td"]), ""])
    if len(data) == 1:
        data.append([Paragraph("-", st["tdc"]), Paragraph("선택된 증거가 없습니다.", st["td"]),
                     "", "", ""])
    t = Table(data, colWidths=widths, repeatRows=1, splitByRow=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_HEAD),
        ("GRID", (0, 0), (-1, -1), 0.6, C_LINE),
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ("VALIGN", (0, 1), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [Paragraph("5. 악성 댓글 증거 일람표", st["h2"]), t, Spacer(1, 8),
            Paragraph(f"※ 총 {len(selected)}건의 악성 댓글이 수집되었습니다.", st["body"])]


def _chunks(text, cpl, max_lines):
    """긴 원문을 표 한 행이 한 쪽을 넘지 않는 크기로 자른다.
    조각을 이어 붙이면 원문과 정확히 같다(글자를 하나도 바꾸지 않음).
    cpl = 한 줄 글자 수 추정, max_lines = 한 행 최대 줄 수."""
    s = "" if text is None else str(text)
    out, start, lines, col = [], 0, 1, 0
    for i, ch in enumerate(s):
        if ch == "\n":
            lines, col = lines + 1, 0
        else:
            col += 1
            if col > cpl:
                lines, col = lines + 1, 1
        if lines > max_lines:
            cut = i + 1 if ch == "\n" else i
            out.append(s[start:cut])
            start = cut
            lines, col = 1, (0 if ch == "\n" else 1)
    out.append(s[start:])
    return [c for c in out if c] or [""]


def _show(chunk):
    """조각 끝의 줄바꿈은 행이 바뀌므로 화면에서만 생략한다."""
    return chunk[:-1] if chunk.endswith("\n") else chunk


def _long_text_rows(text, st):
    """상세 표의 '댓글 원문' 칸. 아주 긴 원문은 여러 행으로 나눈다(글자는 그대로)."""
    return [Paragraph(para_text(_show(c), NA if i == 0 else ""), st["cell"])
            for i, c in enumerate(_chunks(text, 40, 32))]


def _detail(selected, meta, st):
    out = [Paragraph("6. 개별 댓글 상세 자료", st["h2"])]
    if not selected:
        return out + [Paragraph("선택된 증거가 없습니다.", st["body"])]
    for i, r in enumerate(selected):
        if i:
            out.append(PageBreak())
        no = f"EV-{i + 1:03d}"
        out += [band(f"증거번호 {no}", st), Spacer(1, 8)]
        texts = _long_text_rows(r.get("text"), st)
        rows = [("작성자", val(r.get("author"))), ("작성 시각", f"{kst(r.get('time'))} (KST)"
                                                   if _parse_dt(r.get("time")) else NA)]
        rows.append(("댓글 원문" if len(texts) == 1 else "댓글 원문 (1)", texts[0]))
        for k, p in enumerate(texts[1:], 2):
            rows.append((f"댓글 원문 ({k})", p))
        rows += [("댓글 ID", val(r.get("id"))), ("원본 URL", link(r.get("link"), st))]
        if r.get("channel"):
            rows.insert(1, ("작성자 채널", link(r.get("channel"), st)))
        out += [kv_table(rows, st, label_w=30 * mm), Spacer(1, 10)]

        imgs = [f for f in (image_flowable(b) for b in images_for(r, i, meta)) if f]
        if imgs:
            out.append(CondPageBreak(40 * mm))
            out.append(Paragraph("원본 캡처 이미지", st["h2"]))
            cap = []
            for k, im in enumerate(imgs):
                if k:
                    cap.append(Spacer(1, 6))
                cap.append(im)
            out.append(box(cap, bg=colors.white))
            out.append(Paragraph("※ 사용자가 직접 올린 캡처 이미지입니다.", st["tiny"]))
        else:
            out.append(KeepTogether([
                Paragraph("댓글 화면 재구성(실제 캡처 아님)", st["h2"]),
                comment_card(r, st),
                Paragraph("※ 수집된 데이터로 다시 그린 화면이며, 실제 화면 캡처가 아닙니다.",
                          st["tiny"])]))
        out += [Spacer(1, 10), KeepTogether([ai_note(r, st)])]
    return out


def _closing(selected, meta, st):
    src = str(val(meta.get("src"), "")).lower()
    is_yt = "youtube" in src or "유튜브" in src or bool(meta.get("url"))
    method = ("YouTube Data API v3 공식 응답(commentThreads·comments)으로 수집했습니다."
              if is_yt else
              f"사용자가 입력한 텍스트·스크린샷(문자 인식)으로 만들었습니다"
              f"{' (' + str(meta.get('src')) + ')' if meta.get('src') else ''}. "
              "자동 문자 인식은 오류가 있을 수 있어 원본 이미지와 대조가 필요합니다.")
    at = meta.get("collected_at")
    rows = [
        f"수집 방법: {method}",
        f"수집 대상: {val(meta.get('url'))}",
        f"수집 일시: {kst(at, '%Y.%m.%d %H:%M:%S')} (KST) / {utc(at)} (UTC)",
        f"전체 수집 댓글 수: {val(meta.get('total'))}{'건' if meta.get('total') not in (None, '') else ''}"
        f" / 보고서 수록 증거: {len(selected)}건",
        f"원본 데이터 SHA-256: {val(meta.get('sha256'))}",
        "SHA-256은 수집 직후 원본 데이터에서 계산한 값으로, 함께 내려받은 원본 파일에서 같은 값이 "
        "나오면 내용이 바뀌지 않았음을 확인할 수 있습니다.",
        "본 보고서는 경찰청 공식 서식(고소장)이 아니며, 신고 및 고소 준비를 위한 참고자료입니다. "
        "분류 결과와 법적 검토 내용은 자동 분석에 따른 참고 의견이며, 실제 법적 판단은 수사기관과 "
        "법원의 결정에 따릅니다. 원 게시물이 삭제될 수 있으니 원본 URL을 직접 열어 화면을 보존해 "
        "두시기를 권장합니다.",
    ]
    body = [Paragraph("수집 정보 및 유의사항", st["cellb"]), Spacer(1, 4)]
    body += [Paragraph(para_text(x), st["tiny"]) for x in rows]
    return [Spacer(1, 14), CondPageBreak(60 * mm), box(body, pad=8)]


# =====================================================================
# 진입점
# =====================================================================
def make_pdf(selected, meta, case) -> bytes:
    _register_fonts()
    selected = list(selected or [])
    meta = dict(meta or {})
    case = dict(case or {})
    st = _styles()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
        topMargin=29 * mm, bottomMargin=21 * mm,
        title="EVIDA 증거 정리 보고서", author="EVIDA",
        subject="신고 및 고소 준비 참고자료",
    )
    story = []
    story += _overview(selected, case, st)
    story.append(PageBreak())
    story += _evidence_table(selected, st)
    story.append(PageBreak())
    story += _detail(selected, meta, st)
    story += _closing(selected, meta, st)
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


# =====================================================================
# 자체 점검: python report.py → sample_report.pdf
# =====================================================================
if __name__ == "__main__":
    import hashlib
    import json

    sample = [
        {"id": "Ugx1abcDEF", "author": "@악플러123", "channel": "https://www.youtube.com/@x",
         "cid": "UC123", "text": "너 같은 병신은 처음 봄 🤬🤬\n진짜 꺼져라",
         "time": "2025-03-02T14:05:00Z",
         "link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&lc=Ugx1abcDEF", "label": "악플",
         "severity": 3, "category": "욕설, 인신공격", "reason": "욕설: ‘병신’ → 상대 지목 표현 있음",
         "law": "모욕 의심", "statute": "형법 제311조", "ai": 97, "gem": "특정인 비하", "memo": ""},
        {"id": "Ugx2", "author": "", "text": "어디 사는지 다 안다 조심해 " * 3,
         "time": "", "link": "", "label": "악플", "severity": 2, "category": "위협",
         "reason": "위협: ‘거주지·동선 언급/경고’", "law": "협박 의심", "statute": "형법 제283조",
         "ai": None, "gem": "", "memo": "같은 계정 반복"},
        {"id": "Ugx3", "author": "긴글러", "text": "가나다라마바사 <b>태그</b> & 기호 " * 300,
         "time": "2025-03-05T01:00:00+00:00", "link": "https://youtu.be/x", "label": "악플",
         "severity": 1, "category": "AI 판단", "reason": "", "law": "", "statute": "", "ai": 70},
    ]
    from PIL import Image as PILImage, ImageDraw
    im = PILImage.new("RGB", (900, 300), "#202124")
    ImageDraw.Draw(im).text((20, 20), "screenshot", fill="white")
    b = io.BytesIO()
    im.save(b, format="PNG")
    raw = json.dumps(sample, ensure_ascii=False).encode()
    meta = {"src": "YouTube", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(), "total": 1234,
            "images": {"Ugx2": b.getvalue()}}
    case = {"title": "유튜브 채널 악성 댓글 피해", "platform": "YouTube",
            "period": (date(2025, 3, 2), date(2025, 3, 5)),
            "story": "영상 업로드 이후 같은 계정들이 반복해서 댓글을 남겼습니다.\n\n둘째 문단입니다.",
            "harm": "정신적 스트레스로 진료를 받았습니다.", "note": ""}
    pdf = make_pdf(sample, meta, case)
    with open("sample_report.pdf", "wb") as f:
        f.write(pdf)
    print(f"sample_report.pdf 생성: {len(pdf):,} bytes")
    pdf2 = make_pdf([], {}, {})
    print(f"빈 보고서도 생성: {len(pdf2):,} bytes")
