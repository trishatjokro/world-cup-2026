"""Match World Cup 2026 squad players to the EA Sports FC (FIFA 23 final) SoFIFA
dataset, producing data/players.csv with face image URLs + attributes.

Source FC dataset: jsulz/FIFA23 male_players.csv (fifa_version=23, update=9),
SoFIFA-derived. Face URLs are cdn.sofifa.net.

Matching strategy
-----------------
For every FC player we build a set of romanised name TOKENS taken from both the
short_name ("L. Messi") and the long_name ("Lionel Andres Messi Cuccittini"),
dropping single-letter initials and any CJK characters. We do the same for each
squad player. A candidate is scored by token-set fuzzy ratio plus a hard guard:
the discriminating tokens (surname; and for short two-token names also the other
token) must actually align. This avoids surname-only collisions (e.g. two "Kim",
two "Silva", two "Suzuki") that would otherwise produce confident-looking but
WRONG matches. When nothing clears the bar the player is left unmatched.
"""
import re
import unicodedata
import pandas as pd
from rapidfuzz import fuzz, process

# Map our country names -> FC nationality_name (only where they differ)
COUNTRY_TO_FC = {
    "Cape Verde": "Cape Verde Islands",
    "Curaçao": "Curacao",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    # the rest match 1:1 (Iran, Czech Republic, United States, Saudi Arabia, etc.)
}

OUTFIELD_LABELS = "PAC|SHO|PAS|DRI|DEF|PHY"
GK_LABELS = "DIV|HAN|KIC|REF|SPD|POS"

# Korean / common East-Asian family names: when the *family* name is shared but
# the given names differ, demand an exact given-name match (handled generically
# by the token guard, but we keep the threshold strict for everyone).


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = strip_accents(s).lower()
    s = s.replace("-", " ").replace(".", " ").replace("'", " ")
    # drop any non-ascii (CJK) leftover
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(*names):
    """Multiset of romanised name tokens from one or more name strings,
    dropping single-letter initials (e.g. the "L" in "L. Messi")."""
    toks = []
    for nm in names:
        for t in norm(nm).split():
            if len(t) > 1:
                toks.append(t)
    return toks


def token_str(*names):
    return " ".join(sorted(set(tokens(*names))))


def initial_surname(short_name):
    """For FC short_names like 'W. Endo' / 'H. Son' return (initial, surname).

    Returns (None, None) if the short name is not an 'X. Surname' pattern.
    Used to rescue East-Asian players whose long_name is CJK (no romaji given
    name), so the only romaji signal is the leading initial + family surname.
    """
    if not isinstance(short_name, str):
        return None, None
    m = re.match(r"^\s*([A-Za-z])\.\s+(.+)$", strip_accents(short_name))
    if not m:
        return None, None
    init = m.group(1).lower()
    surname = norm(m.group(2))
    surname = surname.split()[-1] if surname.split() else ""
    return (init, surname) if surname else (None, None)


def main():
    sq = pd.read_csv("data/squads.csv")
    fc = pd.read_csv("data/fc_latest.csv").reset_index(drop=True)

    # Precompute FC token sets / strings.
    fc["tok_set"] = [set(tokens(fc.at[i, "short_name"], fc.at[i, "long_name"]))
                     for i in fc.index]
    fc["tok_str"] = [" ".join(sorted(s)) for s in fc["tok_set"]]
    # For CJK-long-name players, the romaji given name is missing, so capture
    # the (initial, surname) pair from the short_name as a rescue key.
    fc["init_sur"] = [initial_surname(fc.at[i, "short_name"])
                      if len(fc.at[i, "tok_set"]) <= 1 else (None, None)
                      for i in fc.index]

    # Per-nationality choice lists for rapidfuzz (token_str -> row index).
    def build_pool(sub):
        choices, owners = [], []
        for i in sub.index:
            ts = fc.at[i, "tok_str"]
            if ts:
                choices.append(ts)
                owners.append(i)
        return choices, owners

    fc_by_nat = {nat: build_pool(g) for nat, g in fc.groupby("nationality_name")}
    global_pool = build_pool(fc)

    # Per-nationality (initial, surname) -> [row indices] rescue index.
    nat_initsur = {}
    for i in fc.index:
        init, sur = fc.at[i, "init_sur"]
        if init is None:
            continue
        nat = fc.at[i, "nationality_name"]
        nat_initsur.setdefault(nat, {}).setdefault((init, sur), []).append(i)

    # Per-nationality surname -> [(row index, given-name token list)] for the
    # distinctive-surname nickname rescue. Uses the LAST token of the long_name.
    nat_surname = {}
    for i in fc.index:
        nat = fc.at[i, "nationality_name"]
        toks_long = norm(fc.at[i, "long_name"]).split()
        added = set()
        if len(toks_long) >= 2:
            sur, given = toks_long[-1], toks_long[:-1]
            nat_surname.setdefault(nat, {}).setdefault(sur, []).append((i, given))
            added.add(sur)
        # Also index the short_name's last token (catches FC mononyms like
        # 'Grimaldo' whose long_name surname is a different word 'García').
        toks_short = norm(fc.at[i, "short_name"]).split()
        toks_short = [t for t in toks_short if len(t) > 1]  # drop initials
        if toks_short:
            ssur = toks_short[-1]
            if ssur not in added:
                given_s = toks_long[:-1] if len(toks_long) >= 2 else toks_long
                nat_surname.setdefault(nat, {}).setdefault(ssur, []).append((i, given_s))

    out_rows = []
    matched = 0
    for _, p in sq.iterrows():
        country, pos, player = p["country"], p["position"], p["player"]
        p_toks = set(tokens(player))
        p_str = " ".join(sorted(p_toks))

        fc_nat = COUNTRY_TO_FC.get(country, country)

        best_idx, best_conf = match(p_toks, p_str, fc_by_nat.get(fc_nat), fc)
        if (best_idx is None or best_conf < 1.0) and len(p_toks) >= 2:
            # Global fallback ONLY for multi-token names (dual nationals / FC
            # nationality label differs from squad country). Single-token
            # mononyms (Endrick, Marquinhos) are too collision-prone globally
            # and must match within their own nationality.
            gidx, gconf = match(p_toks, p_str, global_pool, fc, strict=True)
            if gidx is not None and gconf >= 1.0 and gconf > best_conf:
                best_idx, best_conf = gidx, gconf

        # Rescue East-Asian players (FC long_name is CJK -> no romaji given name)
        # via UNIQUE (given-initial, surname) within their nationality.
        if best_idx is None or best_conf < 1.0:
            ridx = rescue_initsur(p_toks, nat_initsur.get(fc_nat))
            if ridx is not None:
                best_idx, best_conf = ridx, 1.0

        # Rescue nickname/full-name pairs (e.g. "Dayot Upamecano" ->
        # "Dayotchanculle Oswald Upamecano", "Álex Grimaldo" -> "Alejandro
        # Grimaldo García") via a DISTINCTIVE, UNIQUE surname in the nationality
        # plus given-name initial OR nickname-substring agreement.
        if best_idx is None or best_conf < 1.0:
            ridx = rescue_surname(player, p_toks, nat_surname.get(fc_nat), fc)
            if ridx is not None:
                best_idx, best_conf = ridx, 1.0

        best = fc.loc[best_idx] if (best_idx is not None and best_conf >= 1.0) else None

        row = {
            "country": country, "player": player, "position": pos,
            "club": p["club"], "shirt_number": p["shirt_number"],
            "age": p["age"], "caps": p["caps"],
        }
        is_gk = (pos == "GK")
        row["radar_labels"] = GK_LABELS if is_gk else OUTFIELD_LABELS
        if best is not None:
            matched += 1
            face = best["player_face_url"]
            has = isinstance(face, str) and face.strip() not in ("", "nan")
            row["has_photo"] = bool(has)
            row["face_url"] = face if has else ""
            row["overall"] = int(float(best["overall"])) if pd.notna(best["overall"]) else ""
            if is_gk:
                vals = [best["goalkeeping_diving"], best["goalkeeping_handling"],
                        best["goalkeeping_kicking"], best["goalkeeping_reflexes"],
                        best["goalkeeping_speed"], best["goalkeeping_positioning"]]
            else:
                vals = [best["pace"], best["shooting"], best["passing"],
                        best["dribbling"], best["defending"], best["physic"]]
            for i, v in enumerate(vals, 1):
                row[f"r{i}"] = int(float(v)) if (pd.notna(v) and str(v) != "") else ""
            row["fc_name"] = best["long_name"]
        else:
            row["has_photo"] = False
            row["face_url"] = ""
            row["overall"] = ""
            for i in range(1, 7):
                row[f"r{i}"] = ""
            row["fc_name"] = ""
        out_rows.append(row)

    cols = ["country", "player", "position", "club", "shirt_number", "age", "caps",
            "has_photo", "face_url", "overall", "radar_labels",
            "r1", "r2", "r3", "r4", "r5", "r6", "fc_name"]
    out = pd.DataFrame(out_rows)[cols]
    out.to_csv("data/players.csv", index=False, encoding="utf-8")
    report(out)


def rescue_initsur(p_toks, idx_map):
    """Match a squad player to a UNIQUE FC (initial, surname) entry.

    p_toks is the set of squad name tokens. For each FC surname that appears in
    p_toks, the FC initial must match the first letter of one of the *other*
    squad tokens (works for both 'Given Surname' and Korean 'Surname Given'
    orders). Only accept when exactly one FC entry qualifies (no ambiguity).
    """
    if not idx_map or len(p_toks) < 2:
        return None
    hits = []
    for (init, sur), idxs in idx_map.items():
        if sur not in p_toks:
            continue
        others = [t for t in p_toks if t != sur]
        if any(t[0] == init for t in others):
            hits.extend(idxs)
    # require a single unambiguous FC entry
    return hits[0] if len(hits) == 1 else None


def rescue_surname(player, p_toks, sur_map, fc):
    """Match via a DISTINCTIVE, UNIQUE surname within the nationality.

    Handles nickname<->full-name pairs the token guard rejects. Requires:
      * squad surname (last token, len>=5) appears as exactly ONE FC surname
        bucket with a single entry in this nationality, and
      * the squad given name agrees with the FC given name via shared initial
        OR a nickname substring (e.g. 'alex' in 'alejandro', 'tino' in
        'valentino'), to avoid grabbing an unrelated same-surname player.
    """
    if not sur_map:
        return None
    sq_toks = norm(player).split()
    if len(sq_toks) < 2:
        return None
    sq_sur = sq_toks[-1]
    sq_given = sq_toks[:-1]
    if len(sq_sur) < 5:
        return None  # short surnames are not distinctive enough
    bucket = sur_map.get(sq_sur)
    if not bucket or len(bucket) != 1:
        return None  # must be unambiguous within the nationality
    idx, fc_given = bucket[0]
    if not sq_given or not fc_given:
        return None
    sg, fg = sq_given[0], fc_given[0]
    ok = (sg[0] == fg[0]) or sg.startswith(fg) or fg.startswith(sg) \
        or sg in fg or fg in sg
    return idx if ok else None


def match(p_toks, p_str, pool, fc, strict=False):
    """Return (row_index, confidence) where confidence>=1.0 means accept.

    confidence is token_set_ratio/100 (0..~1.0+) gated by a hard token guard.
    """
    if not pool or not pool[0] or not p_toks:
        return None, 0.0
    choices, owners = pool
    # Get top fuzzy candidates by token-set ratio (order independent).
    cands = process.extract(p_str, choices, scorer=fuzz.token_set_ratio,
                            limit=8, score_cutoff=70)
    best_idx, best_conf = None, 0.0
    for _cstr, score, pos in cands:
        idx = owners[pos]
        f_toks = fc.at[idx, "tok_set"]
        conf = accept_confidence(p_toks, f_toks, score, strict)
        if conf > best_conf:
            best_conf, best_idx = conf, idx
    return best_idx, best_conf


def accept_confidence(p_toks, f_toks, fuzzy_score, strict):
    """Hard guard against surname-only / wrong-given-name collisions.

    Returns >=1.0 only when the squad player's tokens are genuinely covered by
    the FC entry's tokens (and vice versa for the discriminating ones).
    """
    if not p_toks or not f_toks:
        return 0.0
    inter = p_toks & f_toks
    # How many of the squad tokens are present (exact) in the FC token set?
    exact_cover = len(inter)
    # Also allow near-exact token alignment (typo / transliteration variants).
    near = 0
    for pt in p_toks:
        if pt in f_toks:
            continue
        if any(fuzz.ratio(pt, ft) >= 88 for ft in f_toks):
            near += 1
    covered = exact_cover + near

    n_p, n_f = len(p_toks), len(f_toks)
    # Single-token squad names (mononyms: Endrick, Marquinhos, Pedri) must match
    # EXACTLY — fuzzy 'near' matches cause collisions ('endrick'~'hendrick').
    if n_p == 1 and exact_cover < 1:
        return 0.0
    # Require essentially ALL squad tokens to be accounted for. For multi-token
    # names this kills "shared surname only" matches.
    need = n_p if n_p <= 2 else n_p - 1  # tolerate one dropped middle name
    if covered < need:
        return 0.0
    # For two-token names (very common, e.g. "Son Heungmin"), both tokens must
    # align exactly — a single shared token (the surname) is never enough.
    if n_p == 2 and exact_cover + near < 2:
        return 0.0
    # Symmetric check: the FC entry shouldn't have many extra discriminating
    # tokens that the squad name lacks when names are short.
    if strict and n_f - covered > 1 and n_f <= 3:
        return 0.0
    # Confidence: fuzzy score boosted by full coverage.
    base = fuzzy_score / 100.0
    if covered >= n_p:
        base = max(base, 1.0 + 0.01 * covered)
    return base if base >= 1.0 else 0.0


def report(out):
    total = len(out)
    with_face = int(out["has_photo"].sum())
    print(f"Total rows: {total} (expected 1245)")
    print(f"Matched with face_url: {with_face} ({100*with_face/total:.1f}%)")
    print(f"Matched (overall present): {(out['overall'] != '').sum()}")
    print("\nPer-position face coverage:")
    for pos in ["GK", "DF", "MF", "FW"]:
        sub = out[out["position"] == pos]
        print(f"  {pos}: {int(sub['has_photo'].sum())}/{len(sub)} "
              f"({100*sub['has_photo'].mean():.0f}%)")
    print("\nWorst-covered countries:")
    cov = out.groupby("country")["has_photo"].mean().sort_values()
    for c, v in cov.head(10).items():
        print(f"  {c}: {100*v:.0f}%")
    print("\n8 example matched rows:")
    for _, r in out[out["has_photo"]].head(8).iterrows():
        print(f"  {r['player']:<24} {r['country']:<14} OVR {str(r['overall']):<3} {r['face_url']}")
    print("\n5 unmatched examples:")
    for _, r in out[~out["has_photo"]].head(5).iterrows():
        print(f"  {r['player']:<24} {r['country']:<14} ({r['position']})")
    print("\nStar spot-check:")
    for nm, ctry in [("Messi", "Argentina"), ("Mbappé", "France"),
                     ("Bellingham", "England"), ("Haaland", "Norway"),
                     ("Son Heung", "South Korea")]:
        s = out[(out["country"] == ctry) & (out["player"].str.contains(nm, na=False))]
        for _, r in s.iterrows():
            print(f"  {r['player']:<22} {ctry:<12} OVR {r['overall']} {r['face_url']}")


if __name__ == "__main__":
    main()
