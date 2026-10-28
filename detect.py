"""EVIDA — 악플 후보 분류 (규칙 그물 + 학습형 AI)

classify(text)            → {label, severity, category, reason, law, statute}
analyze(rows)             → rows 각각에 classify 결과를 채워 돌려줌 (memo·gem 등은 유지)
train_model()             → 학습된 sklearn Pipeline 또는 None (st.cache_resource로 1회만 학습)
apply_model(rows, model)  → ai 점수(0~100) 부착, 규칙=정상 & 점수≥65 → "AI 판단" severity 1

원칙: 원문(text)은 절대 바꾸지 않는다. 모든 전처리는 검사용 사본에만 적용한다.
      규칙은 확실한 욕만 확정하고, 애매한 것은 severity 1(의심)로 넓게 잡는다.
"""
import re
from functools import lru_cache

# =====================================================================
# 0. 출처·상수
# =====================================================================
DATA_CREDIT = ("학습 데이터: Smilegate AI UnSmile (CC BY-NC-ND 4.0), "
               "K-MHaS (CC BY-SA 4.0)")
UNSMILE_URL = ("https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/"
               "main/unsmile_train_v1.0.tsv")
KMHAS_URL = "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_train.txt"
AI_THRESHOLD = 65

# =====================================================================
# 1. 키워드 사전 (유형: 심각도)
# =====================================================================
KEYWORDS = {
    "욕설": (3, ["시발", "씨발", "ㅅㅂ", "ㅆㅂ", "씨바", "씨팔", "좆", "ㅈ같", "x같", "개새끼",
                 "새끼", "ㅅㄲ", "병신", "ㅂㅅ", "ㅄ", "븅신", "지랄", "ㅈㄹ", "염병", "썅",
                 "엠창", "느금", "ㄴㄱㅁ", "니애미", "애미", "애비", "18놈", "씹새",
                 "fuck", "shit", "bitch"]),
    "비속어": (1, ["존나", "졸라", "ㅈㄴ", "개같", "뒤진다", "뒤질"]),
    "비난": (1, ["악마", "꼴보기", "퉤", "ㅉㅉ", "존망", "토할", "와꾸", "가증", "면상", "무지성",
                 "발작버튼", "누칼협", "알빠노", "긁혔", "뇌절", "관종", "ㅗㅗ", "ㅁㅊ", "🖕"]),
    "위협": (3, ["죽어라", "죽으세요", "뒤져라", "뒤지세요", "디져", "자살해", "죽여버",
                 "찾아간다", "패버"]),
    "인신공격": (2, ["미친놈", "미친년", "또라이", "멍청", "찐따", "한심", "쓰레기", "못생",
                     "정신병자", "싸이코패스", "양아치", "금수", "모자란", "머저리", "역겹",
                     "꺼져", "꺼지세요", "닥쳐", "닥치세요", "상태봐라", "꼴값", "얼굴값",
                     "능지", "쿨찐", "방구석여포", "돼지같", "오크녀", "개저씨", "빻았",
                     "꽃뱀", "맞을짓", "자작극", "idiot", "stupid", "loser"]),
    "혐오": (3, ["한남", "한녀", "김치녀", "틀딱", "급식충", "잼민", "맘충", "짱깨", "쪽바리",
                 "스윗한남", "퐁퐁남", "영포티"]),
    "성적": (3, ["따먹", "몸매보", "ㅗㅜㅑ", "눈요기", "육덕", "스폰녀", "오피녀", "조건만남"]),
}
# 대상 없이 감탄사처럼 쓰이는 욕
EXCLAIM = {"ㅅㅂ", "ㅆㅂ", "시발", "씨발", "존나", "ㅈㄴ", "ㅈㄹ", "fuck", "shit"}
# 초성 문장 안에서는 무시할 초성 키워드
CHOSEONG_SOFT = {"ㅈㄹ", "ㅂㅅ", "ㅄ"}
# 발음 정규화 사본에서 추가로 찾을 키워드 (정규화형 → 대표형)
PRON_KEYWORDS = {"시발": "시발", "시팔": "씨팔", "병신": "병신", "지랄": "지랄", "십새": "씹새"}
ENGLISH_KW = {k for _, (_, ks) in KEYWORDS.items() for k in ks if re.fullmatch(r"[a-z]+", k)}

# 완화되지 않는 유형
NO_MITIGATE = {"위협", "성적", "명예훼손", "신상노출"}

# =====================================================================
# 2. 법적 검토 표
# =====================================================================
_INSULT = ("모욕 의심", "형법 제311조")
LAW = {
    "욕설": _INSULT, "비속어": _INSULT, "비난": _INSULT, "인신공격": _INSULT,
    "정중한 비하": _INSULT,
    "위협": ("협박 의심", "형법 제283조"),
    "성적": ("성희롱·모욕 의심", "형법 제311조, DM이면 성폭력처벌법 제13조"),
    "명예훼손": ("명예훼손 검토", "정보통신망법 제70조"),
    "혐오": ("집단 비하(개인 특정 어려우면 고소 어려움)", "개인이 특정되면 형법 제311조"),
    "집단 비하": ("집단 비하(개인 특정 어려우면 고소 어려움)", "개인이 특정되면 형법 제311조"),
    "신상노출": ("신상 노출 의심(삭제 요청 권장)", "플랫폼 신고"),
    "AI 판단": ("문맥상 악플 의심(사람 검토 필요)", ""),
}

# =====================================================================
# 3. 오탐 예외 — 검사 전에 지운다
# =====================================================================
EXCEPTIONS = [re.compile(p, re.I) for p in [
    r"https?://\S+", r"www\.\S+",
    r"시발점", r"시발택시", r"시발역", r"시바견", r"시바이누",
    r"\d+\s*년\s*병신년", r"병신년\s*(?=(새해|생|년생|에|의|부터|까지|\(|\d))",
    r"애비뉴", r"금수저", r"금수강산", r"능지처참", r"썅썅바", r"장애물", r"정신병원",
    r"쓰레기\s*통", r"분리수거",
    r"새끼\s*(고양이|강아지|손가락|발가락|줄|돼지|오리|새|곰|양|발톱|냥이?|댕댕이|거북이?|여우|사자|호랑이|토끼|원숭이|고기)",
    r"(딸기|사과|열매|과일|감|복숭아|포도|체리|앵두|산딸기|오디|땅)\s*(를|을)?\s*따먹(기|어|었|고)?",
    r"(불|화면|전원|컴퓨터|컴|폰|핸드폰|휴대폰|티비|tv|조명|전기|시동|모니터|엔진|노트북|방송|라이트|램프|전등)\s*(이|가|은|는|도)?\s*(다\s*)?꺼져",
    r"(위기|시련|추위|태풍|장마|불행|재난|시험|마감|겨울|어려움|고난|위험|불황|한파|폭염)\s*(이|가)?\s*(곧\s*)?닥쳐",
    r"(잠|시간|돈|일손|인력|자금|재료|용량|공간|예산|체력|수면|잠이)\s*(이|가|도)?\s*(좀\s*)?모자란",
    r"주제에\s*(대해|대한|관해|관한|맞)",
    r"한남동", r"한남대교", r"한남역", r"한남더힐", r"한남오거리", r"한남자", r"한남성",
    r"한녀석", r"한녀자", r"뒤질세라", r"조작감", r"조작법", r"조작키", r"조작성",
]]

# 단어 끝 "~충" 중 혐오가 아닌 것
CHUNG_OK = ("대충", "보충", "곤충", "해충", "충분", "충격", "충전", "기생충", "회충", "확충", "유충",
            "성충", "갑충", "익충", "살충", "방충", "구충", "요충", "완충", "절충", "상충", "충충",
            "애국충", "이충", "의충", "단충", "장충", "청충")

# =====================================================================
# 4. 한글 자모 도구
# =====================================================================
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ",
        "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
V_COMB = {("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ", ("ㅜ", "ㅓ"): "ㅝ",
          ("ㅜ", "ㅔ"): "ㅞ", ("ㅜ", "ㅣ"): "ㅟ", ("ㅡ", "ㅣ"): "ㅢ"}
F_COMB = {("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ", ("ㄹ", "ㄱ"): "ㄺ",
          ("ㄹ", "ㅁ"): "ㄻ", ("ㄹ", "ㅂ"): "ㄼ", ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ",
          ("ㄹ", "ㅍ"): "ㄿ", ("ㄹ", "ㅎ"): "ㅀ", ("ㅂ", "ㅅ"): "ㅄ"}
KEYMAP = {"q": "ㅂ", "w": "ㅈ", "e": "ㄷ", "r": "ㄱ", "t": "ㅅ", "y": "ㅛ", "u": "ㅕ", "i": "ㅑ",
          "o": "ㅐ", "p": "ㅔ", "a": "ㅁ", "s": "ㄴ", "d": "ㅇ", "f": "ㄹ", "g": "ㅎ", "h": "ㅗ",
          "j": "ㅓ", "k": "ㅏ", "l": "ㅣ", "z": "ㅋ", "x": "ㅌ", "c": "ㅊ", "v": "ㅍ", "b": "ㅠ",
          "n": "ㅜ", "m": "ㅡ", "Q": "ㅃ", "W": "ㅉ", "E": "ㄸ", "R": "ㄲ", "T": "ㅆ", "O": "ㅒ",
          "P": "ㅖ"}


def _is_syl(ch):
    return "가" <= ch <= "힣"


def _decomp(ch):
    n = ord(ch) - 0xAC00
    return n // 588, (n % 588) // 28, n % 28


def _comp(c, v, f=0):
    return chr(0xAC00 + c * 588 + v * 28 + f)


def eng2kor(chunk):
    """영타 → 한글 (두벌식). 낱자가 남으면 None."""
    jamo = []
    for ch in chunk:
        k = KEYMAP.get(ch) or KEYMAP.get(ch.lower())
        if not k:
            return None
        jamo.append(k)
    out, i, n = [], 0, len(jamo)
    isv = lambda j: j in JUNG
    while i < n:
        c = jamo[i]
        if isv(c) or i + 1 >= n or not isv(jamo[i + 1]):
            return None
        v = jamo[i + 1]
        i += 2
        if i < n and isv(jamo[i]) and (v, jamo[i]) in V_COMB:
            v = V_COMB[(v, jamo[i])]
            i += 1
        f = ""
        if i < n and not isv(jamo[i]) and not (i + 1 < n and isv(jamo[i + 1])):
            f = jamo[i]
            i += 1
            if (i < n and not isv(jamo[i]) and (f, jamo[i]) in F_COMB
                    and not (i + 1 < n and isv(jamo[i + 1]))):
                f = F_COMB[(f, jamo[i])]
                i += 1
        if f not in JONG or (i < n and isv(jamo[i])):
            return None
        out.append(_comp(CHO.index(c), JUNG.index(v), JONG.index(f)))
    return "".join(out) or None


# =====================================================================
# 5. 전처리 (검사용 사본만)
# =====================================================================
ROMA = {"ssibal": "씨발", "sibal": "시발", "shibal": "시발", "ssibbal": "씨발", "sibbal": "씨발",
        "cbal": "씨발", "bal": "발", "si": "시", "ssi": "씨", "shi": "시",
        "byungsin": "병신", "byungshin": "병신", "byeongsin": "병신", "byeongshin": "병신",
        "byung": "병", "byeong": "병", "sin": "신", "shin": "신",
        "gae": "개", "sae": "새", "saekki": "새끼", "kki": "끼", "ggi": "끼",
        "jiral": "지랄", "jirral": "지랄", "jot": "좆", "jonna": "존나", "ssang": "썅", "nom": "놈"}
_ROMA_MAX = max(map(len, ROMA))


def _roma(low):
    """로마자 덩어리를 사전 조각으로 완전히 나눌 수 있으면 한글로, 아니면 None."""
    n = len(low)
    best = [None] * (n + 1)
    best[0] = ""
    for i in range(n):
        if best[i] is None:
            continue
        for L in range(1, _ROMA_MAX + 1):
            piece = low[i:i + L]
            if len(piece) == L and piece in ROMA and best[i + L] is None:
                best[i + L] = best[i] + ROMA[piece]
    return best[n]


def _english_sentence(s):
    words = re.findall(r"[A-Za-z]{2,}", s)
    letters = re.findall(r"[A-Za-z가-힣ㄱ-ㅣ]", s)
    han = re.findall(r"[가-힣ㄱ-ㅣ]", s)
    return len(words) >= 4 and letters and len(han) / len(letters) <= 0.10


def _restore(s):
    """영문 덩어리만 복원: 로마자 사전 → 영타 변환. 영어 문장이면 그대로."""
    if not re.search(r"[A-Za-z]", s) or _english_sentence(s):
        return s
    s = re.sub(r"(?i)(?<![a-z])c\s*bal(?![a-z])", "씨발", s)
    s = re.sub(r"(?i)(?<![a-z0-9])c8(?![a-z0-9])", "씨발", s)
    s = re.sub(r"(?i)(?<![a-z])c(?=\s*[발바팔빨])", "씨", s)

    def one(m):
        chunk = m.group(0)
        low = chunk.lower()
        if any(k in low for k in ENGLISH_KW):
            return chunk
        if len(low) >= 3 or (m.start() > 0 and re.match(r"[가-힣ㄱ-ㅣ]", s[m.start() - 1])):
            r = _roma(low)
            if r:
                return r
        if len(low) >= 2:
            r = eng2kor(low if chunk.isupper() else chunk)
            if r:
                return r
        return chunk

    return re.sub(r"[A-Za-z]+", one, s)


def _compress(s):
    """늘여쓰기 압축: 씨이이이발 → 씨발, 같은 글자 3번 이상 → 1번(숫자 제외)."""
    out = []
    for ch in s:
        if out and _is_syl(ch) and _is_syl(out[-1]):
            c, v, f = _decomp(ch)
            pc, pv, pf = _decomp(out[-1])
            if c == 11 and f == 0 and pf == 0 and v == pv:  # ㅇ+같은 모음
                continue
        out.append(ch)
    return re.sub(r"([^\d])\1{2,}", r"\1", "".join(out))


def _numbers(s):
    s = re.sub(r"(?<!\d)18(?=[놈년새])", "씨발", s)
    return re.sub(r"(?<!\d)10(?=새)", "씹", s)


def _remove_exceptions(s):
    for p in EXCEPTIONS:
        s = p.sub(" ", s)
    return s


_CHO_NORM = {1: 0, 4: 3, 8: 7, 10: 9, 13: 12}          # 된소리 → 예사소리
_JUNG_NORM = {19: 20, 16: 20, 3: 1, 7: 5}               # ㅢ,ㅟ→ㅣ / ㅒ→ㅐ / ㅖ→ㅔ


def _pron(s):
    """발음 정규화: 씌발·쉬발·씨빨 → 시발"""
    out = []
    for ch in s:
        if _is_syl(ch):
            c, v, f = _decomp(ch)
            out.append(_comp(_CHO_NORM.get(c, c), _JUNG_NORM.get(v, v), f))
        else:
            out.append(ch)
    return "".join(out)


# =====================================================================
# 6. 키워드 매칭
# =====================================================================
SEP = r"[\s0-9ㅇㅋㅎ\W_]{0,4}"


def _kw_regex(kw):
    if re.fullmatch(r"[a-z]+", kw):
        return re.compile(r"(?<![a-z])" + re.escape(kw))
    if len(kw) == 1 or re.search(r"[0-9a-z]", kw):
        return re.compile(re.escape(kw))
    return re.compile("(?=(" + SEP.join(re.escape(c) for c in kw) + "))")


TABLE = []  # (kw, cat, sev, regex)
for _cat, (_sev, _kws) in KEYWORDS.items():
    for _kw in _kws:
        TABLE.append((_kw, _cat, _sev, _kw_regex(_kw)))
PRON_TABLE = [(canon, "욕설", 3, _kw_regex(norm)) for norm, canon in PRON_KEYWORDS.items()]


def _choseong_spans(t):
    spans = []
    for m in re.finditer(r"[ㄱ-ㅎ](?:[\sㄱ-ㅎ]*[ㄱ-ㅎ])?", t):
        if len(re.findall(r"[ㄱ-ㅆㅈ-ㅊㅌㅍ]", m.group(0))) >= 6:  # ㅇ·ㅋ·ㅎ 제외하고 셈
            spans.append(m.span())
    return spans


def _match_ok(t, start, end, kw, spans):
    seg = t[start:end]
    if re.search(r"\s", seg):
        before = t[start - 1] if start > 0 else ""
        after = t[end] if end < len(t) else ""
        if _is_syl(before or " ") or _is_syl(after or " "):
            return False
    if kw in CHOSEONG_SOFT and any(a <= start and end <= b for a, b in spans):
        return False
    return True


def _scan(t, table, quick=True):
    hits = []
    spans = _choseong_spans(t)
    for kw, cat, sev, rx in table:
        if quick and (kw[0] not in t or kw[-1] not in t):  # 빠른 사전 거르기
            continue
        for m in rx.finditer(t):
            if m.groups():
                s, e = m.start(1), m.end(1)
            else:
                s, e = m.span()
            if e > s and _match_ok(t, s, e, kw, spans):
                hits.append((cat, sev, kw))
                break
    return hits


# =====================================================================
# 7. 문맥 규칙
# =====================================================================
def _any(t, words):
    return next((w for w in words if re.search(w, t)), None)


GROUP = [r"남자", r"여자", r"남성", r"여성", r"성별", r"지역", r"출신", r"나이대", r"국적",
         r"외국인", r"전라", r"경상", r"충청", r"강원도", r"지방", r"조선족", r"중국인", r"일본인",
         r"흑인", r"동남아", r"\d0대", r"노인", r"늙은이", r"엠지", r"mz", r"여초", r"남초"]
GENERAL = [r"과학이", r"과학임", r"과학이네", r"걸러야", r"거르(면|자|세요)", r"원래\s*그렇",
           r"일반인이랑\s*다른", r"믿을\s*수\s*없", r"믿을수없", r"뻔하", r"하는\s*짓", r"종특",
           r"다\s*그렇", r"역시나?\s*\S+\s*답"]
BODY = [r"몸매", r"핏이", r"가슴", r"엉덩", r"허벅지", r"몸\s*라인", r"골반", r"각선미"]
GAZE = [r"너무\s*야해", r"넘\s*야해", r"(^|[\s,.!~])야(하|해|한)", r"꽉\s*차", r"눈이\s*감",
        r"꼴린", r"꼴리", r"침\s*나와", r"침\s*고"]
POLITE = [r"수준\s*낮", r"무식", r"자격\s*없", r"답이\s*없", r"부끄럽지\s*않", r"아는\s*척",
          r"생각이라는\s*걸", r"자체가\s*문제"]
PLACE = [r"어디\s*사는지", r"사는\s*곳", r"학교\s*앞", r"집\s*앞", r"회사\s*앞", r"출퇴근", r"동선"]
WARN = [r"조심해", r"다\s*안다", r"알고\s*있으니", r"기다려라", r"각오해"]
THREAT = [r"가만\s*(안\s*)?두지\s*않", r"가만\s*안\s*둬", r"두고\s*보자", r"두고보자"]
DEFAME = [r"갑질", r"사기", r"횡령", r"불륜", r"학폭", r"표절", r"조작", r"뒷광고", r"소문",
          r"의혹", r"했대", r"라던데"]
GUESS = [r"것\s*같", r"\?", r"라던데", r"했대", r"카더라", r"아닐까", r"인가", r"인듯", r"듯",
         r"것\s*같은데", r"모르겠"]

TARGET = {"너", "니", "넌", "너는", "너도", "너가", "니가", "너를", "널", "너희", "니들", "너네",
          "니네", "느그", "당신", "당신은", "당신이", "님", "님아", "님은", "님이", "너같은",
          "니같은", "you", "u"}
PRAISE = [r"지린다", r"지리", r"찢었", r"찢어", r"미쳤다", r"미쳤네", r"명곡", r"소름", r"레전드",
          r"맛있", r"웃기", r"웃겨", r"귀엽", r"귀여", r"amazing", r"최고", r"잘한다", r"예쁘", r"멋있"]
QUOTE = [r"라고\s*(했|하|함|그러|말)", r"이라고", r"신고\s*했", r"신고함", r"무슨\s*뜻",
         r"뜻이\s*뭐", r"(?m)^\s*\?\?\s*:"]
SELF = {"나", "난", "나는", "나도", "내가"}
OBJECT = [r"날씨", r"게임", r"버스", r"편집", r"회사", r"지하철", r"서버", r"렉", r"시험", r"과제",
          r"월요일", r"출근", r"인터넷", r"와이파이", r"컴퓨터", r"핸드폰"]


def _context(t):
    hits = []
    g, gen = _any(t, GROUP), _any(t, GENERAL)
    if g and gen:
        hits.append(("집단 비하", 1, "집단 일반화"))
    b, z = _any(t, BODY), _any(t, GAZE)
    if b and z:
        hits.append(("성적", 3, "신체+성적 시선"))
    if _any(t, POLITE):
        hits.append(("정중한 비하", 1, "정중한 비하"))
    if (_any(t, PLACE) and _any(t, WARN)) or _any(t, THREAT):
        hits.append(("위협", 2, "거주지·동선 언급/경고"))
    d = _any(t, DEFAME)
    if d:
        hits.append(("명예훼손", 1, "의혹·소문"))
    if re.search(r"01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}", t) or re.search(r"\d+\s*동\s*\d+\s*호", t):
        hits.append(("신상노출", 3, "전화번호·동호수"))
    elif re.search(r"[가-힣]+(초|중|고|대학교|학교|고등학교|중학교|초등학교)", t) and \
            re.search(r"재학|다니(는|고|던|더라)", t):
        hits.append(("신상노출", 1, "학교·재학 정보"))
    for m in re.finditer(r"[가-힣]*충(?![가-힣])", t):
        w = m.group(0)
        if len(w) >= 2 and not w.endswith(CHUNG_OK):
            hits.append(("혐오", 3, w))
            break
    return hits


# =====================================================================
# 8. classify
# =====================================================================
CAT_ORDER = ["위협", "성적", "신상노출", "욕설", "혐오", "인신공격", "명예훼손", "집단 비하",
             "정중한 비하", "비속어", "비난", "AI 판단"]


def _variants(text):
    s = _remove_exceptions(text)
    s = _compress(s)
    s = _remove_exceptions(s)
    s = _numbers(s)
    base = s.lower()
    rest = _restore(s).lower()
    return base, rest, _pron(rest)


@lru_cache(maxsize=50000)
def _classify(text):
    base, rest, pron = _variants(text or "")
    hits = _scan(base, TABLE)
    if rest != base:
        hits += _scan(rest, TABLE)
    hits += _scan(pron, PRON_TABLE, quick=False)
    hits += _context(base)
    # 중복 제거
    seen, uniq = set(), []
    for h in hits:
        if (h[0], h[2]) not in seen:
            seen.add((h[0], h[2]))
            uniq.append(list(h))
    hits = uniq
    if not hits:
        return _result(0, [], "")

    tokens = set(re.findall(r"[가-힣]+|[a-z]+", base))
    has_target = bool(tokens & TARGET) or bool(re.search(r"이\s*새끼", base))
    mitig = [h for h in hits if h[0] not in NO_MITIGATE]
    fixed = [h for h in hits if h[0] in NO_MITIGATE]
    swear_kws = {h[2] for h in mitig if h[0] in KEYWORDS}
    only_excl = bool(mitig) and all(h[2] in EXCLAIM for h in mitig)
    note = ""

    if not has_target and mitig:
        if only_excl and not fixed and _any(base, PRAISE):
            return _result(0, [], "칭찬·감탄 속 감탄사 욕으로 보여 정상 처리")
        reasons = []
        if only_excl:
            reasons.append("대상 없는 감탄사 욕일 수 있음")
        if _any(base, QUOTE):
            reasons.append("인용·신고·뜻 질문일 수 있음")
        if tokens & SELF or re.search(r"내\s*모습", base):
            reasons.append("자기 비하일 수 있음")
        if _any(base, OBJECT):
            reasons.append("사물·상황 대상일 수 있음")
        if reasons:
            for h in mitig:
                h[1] = min(h[1], 1)
            note = " / ".join(reasons)
    elif has_target and swear_kws:
        note = "상대 지목 표현 있음"

    # 명예훼손: 추측/단정 구분
    if any(h[0] == "명예훼손" for h in hits):
        kind = "추측·전언 표현" if _any(base, GUESS) else "단정 표현"
        note = (note + " / " if note else "") + f"명예훼손 관련 {kind}"
    return _result(max(h[1] for h in hits), hits, note)


def _result(sev, hits, note):
    if sev <= 0:
        return {"label": "정상", "severity": 0, "category": "", "reason": note,
                "law": "", "statute": ""}
    hits = sorted(hits, key=lambda h: (-h[1], CAT_ORDER.index(h[0]) if h[0] in CAT_ORDER else 99))
    cats = list(dict.fromkeys(h[0] for h in hits))
    parts = []
    for c in cats:
        kws = [h[2] for h in hits if h[0] == c][:3]
        parts.append(f"{c}: " + ", ".join(f"‘{k}’" for k in kws))
    reason = " · ".join(parts) + (f" → {note}" if note else "")
    laws = list(dict.fromkeys(LAW[c][0] for c in cats if c in LAW))
    stats = list(dict.fromkeys(LAW[c][1] for c in cats if c in LAW and LAW[c][1]))
    return {"label": "악플", "severity": int(sev), "category": ", ".join(cats), "reason": reason,
            "law": " / ".join(laws), "statute": " / ".join(stats)}


def classify(text):
    """label, severity, category, reason, law, statute"""
    return dict(_classify(str(text) if text is not None else ""))


def analyze(rows):
    """rows(공통 형식)에 규칙 분류 결과를 채운다. 원문·memo·gem은 건드리지 않는다."""
    for r in rows:
        r.update(classify(r.get("text", "")))
    return rows


# =====================================================================
# 9. 학습형 AI
# =====================================================================
def _load_data():
    import pandas as pd
    frames = []
    try:
        u = pd.read_csv(UNSMILE_URL, sep="\t")
        frames.append(pd.DataFrame({"x": u["문장"].astype(str),
                                    "y": (u["clean"].astype(int) == 0).astype(int)}))
    except Exception:
        pass
    try:
        k = pd.read_csv(KMHAS_URL, sep="\t", quoting=3, dtype=str)
        x = k["document"].astype(str).str.strip().str.strip('"')
        lab = k["label"].astype(str).str.strip().str.strip('"')
        frames.append(pd.DataFrame({"x": x, "y": (lab != "8").astype(int)}))
    except Exception:
        pass
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).dropna()
    df = df[df["x"].str.len() > 0]
    return df if df["y"].nunique() == 2 else None


def _train():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        df = _load_data()
        if df is None:
            return None
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3,
                            sublinear_tf=True, max_features=300000),
            LogisticRegression(max_iter=3000, C=4, class_weight="balanced"),
        )
        model.fit(df["x"].tolist(), df["y"].tolist())
        return model
    except Exception:
        return None


try:
    import streamlit as st

    @st.cache_resource(show_spinner="학습형 AI 준비 중… (처음 한 번만)")
    def train_model():
        """서버 시작 후 1번만 학습. 실패하면 None."""
        return _train()
except Exception:  # streamlit 없이 실행할 때
    @lru_cache(maxsize=1)
    def train_model():
        return _train()


def _ai_target(text):
    t = str(text or "")
    syl = len(re.findall(r"[가-힣]", t))
    jamo = len(re.findall(r"[ㄱ-ㅣ]", t))
    return syl >= 2 and jamo <= syl


def apply_model(rows, model):
    """ai 점수 부착. 규칙=정상인데 점수≥65면 'AI 판단' severity 1."""
    if model is None or not rows:
        return rows
    idx = [i for i, r in enumerate(rows) if _ai_target(r.get("text"))]
    for r in rows:
        r["ai"] = None
    if not idx:
        return rows
    probs = model.predict_proba([str(rows[i].get("text", "")) for i in idx])[:, 1]
    for i, p in zip(idx, probs):
        r = rows[i]
        score = int(round(float(p) * 100))
        r["ai"] = score
        if r.get("label") != "악플" and score >= AI_THRESHOLD:
            law, statute = LAW["AI 판단"]
            r.update({"label": "악플", "severity": 1, "category": "AI 판단",
                      "reason": f"학습형 AI 점수 {score}점 — 문맥상 악플 가능성(사람 검토 필요)",
                      "law": law, "statute": statute})
    return rows


# =====================================================================
# 10. 합격 기준 자동 검사
# =====================================================================
if __name__ == "__main__":
    import sys

    MUST_BAD = ["tlqkf 진짜", "시ㅇㅇbal", "씌발 뭐냐", "병1신", "씨ㅋㅋ발", "x같네", "꺼져라 관종아",
                "너 같은 병신은 처음 봄", "씨이이이발", "10새야", "틀딱 소리", "죽어라", "fuck you",
                "꺼지세요", "꽃뱀 아니냐", "핏이 너무 야해서",
                "역시 그 성별 과학이네 지능 수준 잘 보고 갑니다"]
    MUST_OK = ["My name is Tania. I recently lost my job and the CCTV/security field.",
               "질병 신호가 왔다", "그 애 비밀", "새끼 고양이 귀엽다", "시발점이 어디야", "금수저 논란",
               "불이 꺼져서 무서웠어", "사발면 먹고싶다", "2018년에 찍은 영상", "와 노래 지린다 ㅅㅂ",
               "한남동 맛집", "보충 설명 감사해요", "나는 봐야하는데 라인전",
               "ㄴ ㅇㄱ ㄱㅇㅈ ㅇ ㅎㅂ ㅈㅈㄹㅂㄹㄹ"]
    MUST_ONE = ["날씨 시발 너무 춥다", "아 씨발 이건 못참지", "누가 나한테 병신이라고 했어요",
                "갑질 한 것 같은데?"]

    fail = 0

    def check(group, texts, cond):
        global fail
        for t in texts:
            r = classify(t)
            ok = cond(r)
            fail += not ok
            print(f"{'OK  ' if ok else 'FAIL'} [{group}] {t!r} → {r['label']}/{r['severity']} "
                  f"{r['category']} | {r['reason']}")

    check("악플", MUST_BAD, lambda r: r["label"] == "악플")
    check("정상", MUST_OK, lambda r: r["label"] == "정상")
    check("sev1", MUST_ONE, lambda r: r["severity"] == 1)
    total = len(MUST_BAD) + len(MUST_OK) + len(MUST_ONE)
    print(f"\n규칙 검사: {total - fail}/{total} 통과")

    if "--train" in sys.argv:
        import time
        t0 = time.time()
        m = train_model()
        print("학습형 AI:", "실패(None)" if m is None else f"완료 {time.time() - t0:.0f}초")
        if m is not None:
            demo = [{"text": s, **classify(s)} for s in
                    ["얼굴 보니까 밥맛 떨어진다", "오늘 영상 너무 재밌어요", "저런 인간은 사회에서 격리해야 함"]]
            for r in apply_model(demo, m):
                print(r["ai"], r["label"], r["severity"], r["text"])
        print(DATA_CREDIT)
    sys.exit(1 if fail else 0)
