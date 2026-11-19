"""Text normalisation for business names and addresses.

Everything here is deterministic string processing plus small static dictionaries
(legal-form vocabulary, street-type abbreviations, administrative-region names).
Indic-script tokens are transliterated with a lexicon mined from the training
ground truth (see lexicon.py) and an algorithmic fallback.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from unidecode import unidecode

try:  # optional fallback transliteration
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _sans_translit
except Exception:  # pragma: no cover
    sanscript = None
    _sans_translit = None

# ----------------------------------------------------------------------------
# scripts
# ----------------------------------------------------------------------------
_INDIC_RE = re.compile(r"[\u0900-\u0DFF]")
_INDIC_TOKEN_RE = re.compile(r"[\u0900-\u0DFF]+")
_SCRIPT_RANGES = [
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"), (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
]


def indic_script(tok: str) -> Optional[str]:
    for ch in tok:
        o = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= o <= hi:
                return name
    return None


def _fallback_translit(tok: str) -> str:
    """Scheme-based transliteration followed by aggressive simplification."""
    scheme = indic_script(tok)
    out = ""
    if _sans_translit is not None and scheme is not None:
        try:
            out = _sans_translit(tok, getattr(sanscript, scheme.upper()), sanscript.ITRANS)
        except Exception:
            out = ""
    if not out:
        out = unidecode(tok)
    out = out.lower()
    out = re.sub(r"[^a-z]", "", out)
    out = re.sub(r"(.)\1+", r"\1", out)          # collapse doubled letters (aa -> a)
    out = re.sub(r"a$", "", out) if len(out) > 3 else out  # drop schwa
    return out


class Transliterator:
    """Token-level transliteration: lexicon first, algorithmic fallback second."""

    def __init__(self, lexicon: Optional[Dict[str, str]] = None):
        self.lex: Dict[str, str] = dict(lexicon or {})
        self.cache: Dict[str, str] = {}

    def token(self, tok: str) -> str:
        v = self.lex.get(tok)
        if v is not None:
            return v
        v = self.cache.get(tok)
        if v is None:
            v = _fallback_translit(tok)
            self.cache[tok] = v
        return v

    def text(self, s: str) -> str:
        if not _INDIC_RE.search(s):
            return s
        return _INDIC_TOKEN_RE.sub(lambda m: " " + self.token(m.group(0)) + " ", s)


def load_lexicon(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ----------------------------------------------------------------------------
# generic cleaning
# ----------------------------------------------------------------------------
_NULL_TOKENS = {"null", "none", "nil", "na", "n/a", "nan", "<null>", "unknown", "notavailable", "notapplicable", "-", "--"}
_WS_RE = re.compile(r"\s+")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th|er|eme|e)\b")
_PHONE_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
_DIGIT_ALPHA_RE = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")


def fold(s: str) -> str:
    """Unicode fold: NFKC, strip accents / transliterate remaining non-ASCII, lowercase."""
    s = unicodedata.normalize("NFKC", s)
    if not s.isascii():
        s = unidecode(s)
    return s.lower()


def _strip_nulls(s: str) -> str:
    parts = []
    for p in re.split(r"[,;|]", s):
        q = p.strip()
        if q.replace(" ", "").lower() in _NULL_TOKENS or q == "":
            continue
        parts.append(q)
    return ", ".join(parts)


# ----------------------------------------------------------------------------
# names
# ----------------------------------------------------------------------------
LEGAL_MAP = {
    "llc": "llc", "l.l.c": "llc", "pllc": "llc", "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "llp": "llp", "lp": "lp", "plc": "plc", "pa": "pa", "pc": "pc", "md": "md",
    "dds": "dds", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "gmbh": "gmbh", "ag": "ag", "opc": "opc", "sp": "sp",
    "association": "assoc", "assoc": "assoc", "assn": "assoc", "trust": "trust", "foundation": "foundation",
    "lc": "llc", "ltda": "ltda", "sarlu": "sarl", "scop": "scop", "sccv": "sccv", "gie": "gie", "sem": "sem",
    # further French legal designators (pure vocabulary; only tokens that are not ordinary words elsewhere)
    "ei": "ei", "eirl": "eirl", "selas": "selas", "selafa": "selafa", "sca": "sca", "scs": "scs", "scm": "scm",
    "scea": "scea", "gaec": "gaec",
}
LEGAL_TOKENS = set(LEGAL_MAP.values()) | set(LEGAL_MAP.keys())
_FILLER = {"mr", "mrs", "ms", "m/s", "ms.", "messrs", "sri", "shri", "smt", "the", "and", "of", "de", "du", "des",
           "la", "le", "les", "et", "dba", "dr"}
_TLDS = r"(?:com|net|org|in|co\.in|io|biz|info|us|fr|co|edu|gov|me|shop|store|online|site)"
_TLD_RE = re.compile(r"\." + _TLDS + r"\b")
_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9.-]*\." + _TLDS + r"(?:/\S*)?")
_ID_NOISE_RE = re.compile(r"\(?\bid\s*[:#-]?\s*\d+\)?|#\s*\d+\b")
_DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|t/a|trading as|aka|a\.k\.a\.?)\b")
_NAME_KEEP_RE = re.compile(r"[^a-z0-9 ]")


def _name_tokens(s: str) -> List[str]:
    from segment import leet_fix
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s = re.sub(r"(?<=\w)'s\b", "s", s)               # ibarra's -> ibarras
    s = s.replace("'", "").replace("’", "")
    s = _PHONE_RE.sub(" ", s)
    s = _NAME_KEEP_RE.sub(" ", s)
    toks = s.split()
    out = []
    for t in toks:
        if t in _NULL_TOKENS:
            continue
        t = leet_fix(t)
        out.append(LEGAL_MAP.get(t, t))
    return out


def normalize_name(raw: str, tr: Transliterator, seg=None) -> Tuple[str, str, str, str, int, int, int]:
    """Return (name, name_core, name_alias, name_nospace, is_domain, is_nonlatin, has_alias).

    ``seg`` is an optional word :class:`segment.Segmenter` used to split concatenated
    web-domain style names (``hospitalunitedprivate.com``) into words.
    """
    s = raw.strip()
    is_nonlatin = 1 if _INDIC_RE.search(s) else 0
    s = tr.text(s)
    s = fold(s)
    s = _strip_nulls(s)
    s = _ID_NOISE_RE.sub(" ", s)                     # "(id: 82145)", "#24831"
    s = re.sub(r"\bm/s\.?(?=\s|$)", " ", s)           # "M/s Acme" -> "Acme"
    is_domain = 0
    urls = _URL_RE.findall(s)
    if urls:
        without = _URL_RE.sub(" ", s)
        if re.search(r"[a-z]{2,}", without):
            s = without                              # "acme traders | www.acme.com" -> keep the words only
        else:
            is_domain = 1
            core_url = max(urls, key=len)
            core_url = re.sub(r"^(?:https?://)?(?:www\.)?", "", core_url)
            core_url = re.sub(r"/.*$", "", core_url)
            core_url = _TLD_RE.sub(" ", core_url).replace(".", " ").replace("-", " ").strip()
            s = core_url
    elif "@" in s:
        # social-media handle style: "@acmetraders", "smt @acme" -> concatenated words
        handles = re.findall(r"@\s*([a-z0-9_.-]{4,})", s)
        if handles:
            is_domain = 1
            s = re.sub(r"@\s*[a-z0-9_.-]{4,}", lambda m: " " + m.group(0).lstrip("@ ").replace("_", " ").replace("-", " ") + " ", s)
    # dotted acronyms with or without the final dot: "p.a." -> pa, "l.l.c." -> llc, "S.A.S" -> sas
    s = re.sub(r"\b((?:[a-z]\.){1,}[a-z])\b\.?", lambda m: m.group(1).replace(".", "") + " ", s)
    s = re.sub(r"\.(?=[a-z])", " ", s)
    alias = ""
    has_alias = 0
    m = _DBA_RE.search(s)
    if m and _NAME_KEEP_RE.sub("", s[: m.start()]).strip() and _NAME_KEEP_RE.sub("", s[m.end():]).strip():
        # "<primary name> dba <trade name>": both parts must be non-empty, otherwise the
        # marker is just an ordinary token of the name (e.g. "DBA Construction Pvt Ltd")
        alias = s[m.end():]
        s = s[: m.start()]
        has_alias = 1
    toks = _name_tokens(s)
    if is_domain and seg is not None:
        # split concatenated words: "hospitalunitedprivate" -> "hospital united private"
        toks = [w for t in toks for w in (seg.segment(t).split() if t.isalpha() and len(t) >= 6 else [t])]
        toks = [LEGAL_MAP.get(t, t) for t in toks]
    name = " ".join(toks)
    core_toks = [t for t in toks if t not in LEGAL_TOKENS and t not in _FILLER]
    if not core_toks:
        core_toks = [t for t in toks if t not in _FILLER] or toks
    core = " ".join(core_toks)
    alias_core = ""
    if has_alias:
        at = [t for t in _name_tokens(alias) if t not in LEGAL_TOKENS and t not in _FILLER]
        alias_core = " ".join(at)
    nospace = core.replace(" ", "")
    return name, core, alias_core, nospace, is_domain, is_nonlatin, has_alias


# ----------------------------------------------------------------------------
# addresses: static administrative-region dictionaries
# ----------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr",
}
US_CODES = set(US_STATES.values())

IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la", "andaman and nicobar islands": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
}
IN_CODES = {"ap", "ar", "as", "br", "cg", "ga", "gj", "hr", "hp", "jh", "ka", "kl", "mp", "mh", "mn", "ml", "mz",
            "nl", "od", "or", "pb", "rj", "sk", "tn", "ts", "tg", "tr", "up", "uk", "ua", "wb", "dl", "ch", "py", "jk", "la", "an", "dn", "dd", "ld"}
IN_CODE_ALIAS = {"or": "od", "tg": "ts", "ua": "uk"}
# Indic-script state names (the complete set that occurs in the data)
IN_INDIC_STATES = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn", "ગુજરાત": "gj",
    "পশ্চিমবঙ্গ": "wb", "తెలంగాణ": "ts", "हरियाणा": "hr", "राजस्थान": "rj", "കേരളം": "kl", "बिहार": "br",
    "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb", "ଓଡ଼ିଶା": "od", "ओडिशा": "od", "छत्तीसगढ़": "cg",
    "झारखंड": "jh", "उत्तराखंड": "uk", "असम": "as", "গোয়া": "ga", "गोवा": "ga", "हिमाचल प्रदेश": "hp",
}

FR_REGIONS = {
    "auvergne rhone alpes": "ara", "bourgogne franche comte": "bfc", "bretagne": "bre", "centre val de loire": "cvl",
    "corse": "cor", "grand est": "ges", "hauts de france": "hdf", "ile de france": "idf", "normandie": "nor",
    "nouvelle aquitaine": "naq", "occitanie": "occ", "pays de la loire": "pdl", "provence alpes cote d azur": "pac",
    "paca": "pac", "guadeloupe": "gua", "martinique": "mar", "guyane": "guy", "la reunion": "reu", "reunion": "reu", "mayotte": "may",
}
FR_DEPARTMENTS = {
    "ain": "ara", "allier": "ara", "ardeche": "ara", "cantal": "ara", "drome": "ara", "isere": "ara", "loire": "ara",
    "haute loire": "ara", "puy de dome": "ara", "rhone": "ara", "savoie": "ara", "haute savoie": "ara", "metropole de lyon": "ara",
    "cote d or": "bfc", "doubs": "bfc", "jura": "bfc", "nievre": "bfc", "haute saone": "bfc", "saone et loire": "bfc",
    "yonne": "bfc", "territoire de belfort": "bfc",
    "cotes d armor": "bre", "finistere": "bre", "ille et vilaine": "bre", "morbihan": "bre",
    "cher": "cvl", "eure et loir": "cvl", "indre": "cvl", "indre et loire": "cvl", "loir et cher": "cvl", "loiret": "cvl",
    "corse du sud": "cor", "haute corse": "cor",
    "ardennes": "ges", "aube": "ges", "marne": "ges", "haute marne": "ges", "meurthe et moselle": "ges", "meuse": "ges",
    "moselle": "ges", "bas rhin": "ges", "haut rhin": "ges", "vosges": "ges",
    "aisne": "hdf", "nord": "hdf", "oise": "hdf", "pas de calais": "hdf", "somme": "hdf",
    "paris": "idf", "seine et marne": "idf", "yvelines": "idf", "essonne": "idf", "hauts de seine": "idf",
    "seine saint denis": "idf", "val de marne": "idf", "val d oise": "idf",
    "calvados": "nor", "eure": "nor", "manche": "nor", "orne": "nor", "seine maritime": "nor",
    "charente": "naq", "charente maritime": "naq", "correze": "naq", "creuse": "naq", "dordogne": "naq", "gironde": "naq",
    "landes": "naq", "lot et garonne": "naq", "pyrenees atlantiques": "naq", "deux sevres": "naq", "vienne": "naq", "haute vienne": "naq",
    "ariege": "occ", "aude": "occ", "aveyron": "occ", "gard": "occ", "haute garonne": "occ", "gers": "occ", "herault": "occ",
    "lot": "occ", "lozere": "occ", "hautes pyrenees": "occ", "pyrenees orientales": "occ", "tarn": "occ", "tarn et garonne": "occ",
    "loire atlantique": "pdl", "maine et loire": "pdl", "mayenne": "pdl", "sarthe": "pdl", "vendee": "pdl",
    "alpes de haute provence": "pac", "hautes alpes": "pac", "alpes maritimes": "pac", "bouches du rhone": "pac", "var": "pac", "vaucluse": "pac",
}

# street-type / address abbreviations (token level, after folding)
ABBR_COMMON = {
    "ave": "avenue", "av": "avenue", "avn": "avenue", "rd": "road", "dr": "drive", "drv": "drive", "ln": "lane",
    "blvd": "boulevard", "bvd": "boulevard", "ct": "court", "crt": "court", "cir": "circle", "pl": "place", "ter": "terrace",
    "terr": "terrace", "trl": "trail", "pkwy": "parkway", "pky": "parkway", "hwy": "highway", "fwy": "freeway",
    "expy": "expressway", "sq": "square", "plz": "plaza", "pt": "point", "rte": "route", "mt": "mount", "ft": "fort",
    "ste": "suite", "apt": "apartment", "bldg": "building", "fl": "floor", "flr": "floor", "twp": "township",
    "hts": "heights", "jct": "junction", "cv": "cove", "xing": "crossing", "byp": "bypass", "ctr": "center",
    "cntr": "center", "centre": "center", "mnr": "manor", "hollow": "hollow", "holw": "hollow", "lk": "lake", "vly": "valley",
    "vw": "view", "wy": "way", "aly": "alley", "anx": "annex", "bch": "beach", "bnd": "bend", "blf": "bluff",
    "brg": "bridge", "brk": "brook", "cyn": "canyon", "cswy": "causeway", "clf": "cliff", "cor": "corner", "crk": "creek",
    "cres": "crescent", "xrd": "crossroad", "est": "estate", "ests": "estates", "ext": "extension", "extn": "extension",
    "fls": "falls", "fld": "field", "flds": "fields", "frst": "forest", "gdn": "garden", "gdns": "gardens",
    "gtwy": "gateway", "gln": "glen", "grn": "green", "grv": "grove", "hbr": "harbor", "hvn": "haven", "hl": "hill",
    "hls": "hills", "is": "island", "knl": "knoll", "lndg": "landing", "ldg": "lodge", "mdw": "meadow", "mdws": "meadows",
    "ml": "mill", "mtn": "mountain", "orch": "orchard", "pne": "pine", "pnes": "pines", "prt": "port", "rdg": "ridge",
    "riv": "river", "shr": "shore", "spg": "spring", "spgs": "springs", "sta": "station", "smt": "summit", "trce": "trace",
    "tpke": "turnpike", "vlg": "village", "vw": "view", "wls": "wells", "no": "no", "nr": "near", "opp": "opposite",
    "mkt": "market", "ind": "industrial", "indl": "industrial", "soc": "society", "sec": "sector", "sect": "sector",
    "blk": "block", "ph": "phase", "clny": "colony", "nagr": "nagar", "ngr": "nagar", "twn": "town", "vil": "village",
    "dist": "district", "distt": "district", "tal": "taluka", "tq": "taluka", "teh": "tehsil", "po": "po", "ps": "ps",
    "pkt": "pocket", "encl": "enclave", "cmplx": "complex", "complx": "complex", "hosp": "hospital", "sch": "school",
    "stn": "station", "rly": "railway", "gali": "gali", "chk": "chowk", "chowk": "chowk", "apts": "apartments",
    "appt": "apartment", "appts": "apartments", "bhavan": "bhawan", "mandir": "mandir", "univ": "university",
}
ABBR_US = {"st": "street", "str": "street", "sts": "streets", "cres": "crescent", "pk": "park", "pkg": "parking",
           "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"}
ABBR_IN = {"st": "street", "str": "street", "mg": "mg", "pk": "park"}
ABBR_FR = {
    "r": "rue", "rue": "rue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard", "all": "allee", "alle": "allee",
    "allee": "allee", "imp": "impasse", "che": "chemin", "ch": "chemin", "chem": "chemin", "crs": "cours", "prom": "promenade",
    "pas": "passage", "res": "residence", "lot": "lotissement", "lotis": "lotissement", "st": "saint", "ste": "sainte",
    "fbg": "faubourg", "gal": "galerie", "ham": "hameau", "mte": "montee", "pte": "porte", "rpt": "rond point", "sen": "sentier",
    "vla": "villa", "qu": "quai", "esp": "esplanade", "pl": "place", "sq": "square", "rte": "route", "av": "avenue",
    "ave": "avenue", "bat": "batiment", "bt": "batiment", "zi": "zi", "za": "za", "zac": "zac", "cite": "cite", "cd": "chemin departemental",
    "rn": "route nationale", "rd": "route departementale", "grande": "grande", "gde": "grande", "gd": "grand", "pt": "petit", "pte": "petite",
    "vieux": "vieux", "vx": "vieux", "immeuble": "immeuble", "imm": "immeuble", "app": "appartement", "appt": "appartement",
    "etg": "etage", "et": "et", "d": "d", "l": "l",
    # honorifics that dominate French street names ("avenue du General de Gaulle", "rue du Docteur Roux")
    "gal": "general", "gen": "general", "mal": "marechal", "pdt": "president", "cdt": "commandant",
    "lt": "lieutenant", "cne": "capitaine", "cpt": "capitaine", "dr": "docteur", "prof": "professeur",
    "pr": "professeur", "abbe": "abbe", "fg": "faubourg", "ctre": "centre", "bvd": "boulevard", "hlm": "hlm",
    "ancienne": "ancienne", "anc": "ancienne", "nlle": "nouvelle", "nle": "nouvelle", "ste": "sainte", "sts": "saints",
    "pl": "place", "car": "carrefour", "rle": "ruelle", "sent": "sentier", "tsse": "terrasse", "tr": "traverse",
    "vc": "voie communale", "vla": "villa", "pce": "place", "ld": "lieu dit", "cc": "centre commercial",
}
# tokens that only announce a number ("h.no 51", "door no 1", "plot no 4")
_NUMBER_PREFIX = {"no", "h", "hno", "hn", "door", "d", "khasra", "kh", "sy", "survey", "plot", "flat", "shop", "gala",
                  "office", "house", "unit", "num", "number", "nos", "pmb", "ste", "suite", "apt", "apartment", "bis", "ter"}
_ADDR_KEEP_RE = re.compile(r"[^a-z0-9 ]")


def _country_maps(country: str):
    c = (country or "").strip().lower()
    if c in ("us", "usa", "united states", "united states of america"):
        return "us", US_STATES, US_CODES, ABBR_US, {}
    if c in ("india", "in"):
        return "in", IN_STATES, IN_CODES, ABBR_IN, IN_CODE_ALIAS
    if c in ("france", "fr"):
        return "fr", {**FR_REGIONS, **FR_DEPARTMENTS}, set(FR_REGIONS.values()), ABBR_FR, {}
    # unknown country: use the union of dictionaries (region codes may collide but stay consistent)
    return "xx", {**US_STATES, **IN_STATES, **FR_REGIONS, **FR_DEPARTMENTS}, US_CODES | IN_CODES | set(FR_REGIONS.values()), {**ABBR_US}, IN_CODE_ALIAS


def _comp_tokens(comp: str, abbr_c: Dict[str, str]) -> List[str]:
    comp = _ORDINAL_RE.sub(r"\1", comp)
    comp = comp.replace("&", " and ").replace("'", " ").replace("’", " ")
    comp = _ADDR_KEEP_RE.sub(" ", comp)
    comp = _DIGIT_ALPHA_RE.sub(" ", comp)
    toks = []
    for t in comp.split():
        if t in _NULL_TOKENS:
            continue
        t = abbr_c.get(t, ABBR_COMMON.get(t, t))
        if t.isdigit():
            t = t.lstrip("0") or "0"
        toks.append(t)
    return toks


def normalize_address(raw: str, tr: Transliterator, country: str) -> Tuple[str, str, str, str, str, str]:
    """Return (addr, addr_street, addr_region, house_no, addr_nums, addr_rest)."""
    s = raw.strip()
    if not s:
        return "", "", "", "", "", ""
    cc, regions, codes, abbr_c, code_alias = _country_maps(country)
    # Indic state names first (before folding destroys them)
    if _INDIC_RE.search(s):
        for k, v in IN_INDIC_STATES.items():
            if k in s:
                s = s.replace(k, " " + v.upper() + " ")
        s = tr.text(s)
    s = fold(s)
    s = _strip_nulls(s)
    comps = [c.strip() for c in s.split(",") if c.strip()]
    region = ""
    kept: List[List[str]] = []
    for comp in comps:
        ctoks = _comp_tokens(comp, abbr_c)
        if not ctoks:
            continue
        joined = " ".join(ctoks)
        # region detection: whole component is a state/region name or code
        if joined in regions:
            region = regions[joined]
            continue
        if len(ctoks) == 1 and joined in codes:
            region = code_alias.get(joined, joined)
            continue
        if cc == "fr":
            # component may start with the region and continue ("nord 59000")
            if joined in FR_DEPARTMENTS:
                region = FR_DEPARTMENTS[joined]
                continue
        kept.append(ctoks)
    # street component: first component that contains a digit, else first component
    street_idx = -1
    for i, ctoks in enumerate(kept):
        if any(t.isdigit() for t in ctoks):
            street_idx = i
            break
    if street_idx < 0 and kept:
        street_idx = 0
    street_toks = kept[street_idx] if street_idx >= 0 else []
    all_toks: List[str] = [t for ctoks in kept for t in ctoks]
    nums = sorted({t for t in all_toks if t.isdigit()}, key=lambda x: (len(x), x))
    house = ""
    for t in street_toks:
        if t.isdigit():
            house = t
            break
    addr_toks = [t for t in all_toks if t not in _NUMBER_PREFIX]
    addr = " ".join(addr_toks)
    street = " ".join(t for t in street_toks if t not in _NUMBER_PREFIX)
    rest = " ".join(t for t in addr_toks if not t.isdigit())
    return addr, street, region, house, " ".join(nums), rest


NAME_COLS = ["name", "name_core", "name_alias", "name_nospace", "is_domain", "is_nonlatin", "has_alias"]
ADDR_COLS = ["addr", "addr_street", "addr_region", "house_no", "addr_nums", "addr_rest"]


def normalize_frame(df: pd.DataFrame, tr: Transliterator, seg=None) -> pd.DataFrame:
    """Add normalised columns to a source frame (entity_id, business_name, business_address, country)."""
    names = [normalize_name(x, tr, seg) for x in df["business_name"].tolist()]
    addrs = [normalize_address(x, tr, c) for x, c in zip(df["business_address"].tolist(), df["country"].tolist())]
    out = df.copy()
    for i, c in enumerate(NAME_COLS):
        out[c] = [n[i] for n in names]
    for i, c in enumerate(ADDR_COLS):
        out[c] = [a[i] for a in addrs]
    out["addr_empty"] = (out["addr"] == "").astype(np.int8)
    for c in ["is_domain", "is_nonlatin", "has_alias"]:
        out[c] = out[c].astype(np.int8)
    return out
