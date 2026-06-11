"""Build groups.csv and fixtures.csv for the REAL 2026 FIFA World Cup.

Sources: Wikipedia "2026 FIFA World Cup draw" / "...knockout stage" and ESPN schedule.
Team names normalized to the martj42 results.csv convention.
"""
import pandas as pd

# --- Normalization map: source name -> results.csv canonical name ---
NORM = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
}
def n(name):
    return NORM.get(name, name)

# --- Confederations ---
CONF = {
    "Mexico": "CONCACAF", "Canada": "CONCACAF", "United States": "CONCACAF",
    "Haiti": "CONCACAF", "Panama": "CONCACAF",
    "South Africa": "CAF", "Morocco": "CAF", "Ivory Coast": "CAF",
    "Tunisia": "CAF", "Egypt": "CAF", "Cape Verde": "CAF", "Senegal": "CAF",
    "Algeria": "CAF", "DR Congo": "CAF", "Ghana": "CAF",
    "South Korea": "AFC", "Qatar": "AFC", "Australia": "AFC", "Japan": "AFC",
    "Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC",
    "Czech Republic": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Switzerland": "UEFA", "Scotland": "UEFA", "Turkey": "UEFA",
    "Germany": "UEFA", "Netherlands": "UEFA", "Sweden": "UEFA",
    "Belgium": "UEFA", "Spain": "UEFA", "France": "UEFA", "Norway": "UEFA",
    "Austria": "UEFA", "Portugal": "UEFA", "Colombia": "CONMEBOL",
    "England": "UEFA", "Croatia": "UEFA",
    "Brazil": "CONMEBOL", "Paraguay": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Uruguay": "CONMEBOL", "Argentina": "CONMEBOL",
    "New Zealand": "OFC", "Curaçao": "CONCACAF",
}
HOSTS = {"United States", "Canada", "Mexico"}

# --- Groups (raw source names; normalized below) ---
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# === FILE 1: groups.csv ===
grows = []
for g, teams in GROUPS.items():
    for t in teams:
        ct = n(t)
        grows.append({
            "group": g, "team": ct,
            "confederation": CONF[ct],
            "is_host": ct in HOSTS,
        })
groups_df = pd.DataFrame(grows, columns=["group", "team", "confederation", "is_host"])
groups_df.to_csv("data/groups.csv", index=False)

# === FILE 2: fixtures.csv ===
# Group matches: (id, date, group, home, away, city, venue)
G = [
    (1,  "2026-06-11", "A", "Mexico", "South Africa", "Estadio Azteca", "Mexico City"),
    (2,  "2026-06-11", "A", "South Korea", "Czechia", "Estadio Akron", "Zapopan"),
    (3,  "2026-06-12", "B", "Canada", "Bosnia and Herzegovina", "BMO Field", "Toronto"),
    (4,  "2026-06-12", "D", "United States", "Paraguay", "SoFi Stadium", "Inglewood"),
    (5,  "2026-06-13", "B", "Qatar", "Switzerland", "Levi's Stadium", "Santa Clara"),
    (6,  "2026-06-13", "C", "Brazil", "Morocco", "MetLife Stadium", "East Rutherford"),
    (7,  "2026-06-13", "C", "Haiti", "Scotland", "Gillette Stadium", "Foxborough"),
    (8,  "2026-06-13", "D", "Australia", "Türkiye", "BC Place", "Vancouver"),
    (9,  "2026-06-14", "E", "Germany", "Curaçao", "NRG Stadium", "Houston"),
    (10, "2026-06-14", "F", "Netherlands", "Japan", "AT&T Stadium", "Arlington"),
    (11, "2026-06-14", "E", "Ivory Coast", "Ecuador", "Lincoln Financial Field", "Philadelphia"),
    (12, "2026-06-14", "F", "Sweden", "Tunisia", "Estadio BBVA", "Guadalupe"),
    (13, "2026-06-15", "H", "Spain", "Cape Verde", "Mercedes-Benz Stadium", "Atlanta"),
    (14, "2026-06-15", "G", "Belgium", "Egypt", "Lumen Field", "Seattle"),
    (15, "2026-06-15", "H", "Saudi Arabia", "Uruguay", "Hard Rock Stadium", "Miami Gardens"),
    (16, "2026-06-15", "G", "Iran", "New Zealand", "SoFi Stadium", "Inglewood"),
    (17, "2026-06-16", "I", "France", "Senegal", "MetLife Stadium", "East Rutherford"),
    (18, "2026-06-16", "I", "Iraq", "Norway", "Gillette Stadium", "Foxborough"),
    (19, "2026-06-16", "J", "Argentina", "Algeria", "Arrowhead Stadium", "Kansas City"),
    (20, "2026-06-16", "J", "Austria", "Jordan", "Levi's Stadium", "Santa Clara"),
    (21, "2026-06-17", "K", "Portugal", "DR Congo", "NRG Stadium", "Houston"),
    (22, "2026-06-17", "L", "England", "Croatia", "AT&T Stadium", "Arlington"),
    (23, "2026-06-17", "L", "Ghana", "Panama", "BMO Field", "Toronto"),
    (24, "2026-06-17", "K", "Uzbekistan", "Colombia", "Estadio Azteca", "Mexico City"),
    (25, "2026-06-18", "A", "Czechia", "South Africa", "Mercedes-Benz Stadium", "Atlanta"),
    (26, "2026-06-18", "B", "Switzerland", "Bosnia and Herzegovina", "SoFi Stadium", "Inglewood"),
    (27, "2026-06-18", "B", "Canada", "Qatar", "BC Place", "Vancouver"),
    (28, "2026-06-18", "A", "Mexico", "South Korea", "Estadio Akron", "Zapopan"),
    (29, "2026-06-19", "D", "United States", "Australia", "Lumen Field", "Seattle"),
    (30, "2026-06-19", "C", "Scotland", "Morocco", "Gillette Stadium", "Foxborough"),
    (31, "2026-06-19", "C", "Brazil", "Haiti", "Lincoln Financial Field", "Philadelphia"),
    (32, "2026-06-19", "D", "Türkiye", "Paraguay", "Levi's Stadium", "Santa Clara"),
    (33, "2026-06-20", "F", "Netherlands", "Sweden", "NRG Stadium", "Houston"),
    (34, "2026-06-20", "E", "Germany", "Ivory Coast", "BMO Field", "Toronto"),
    (35, "2026-06-20", "E", "Ecuador", "Curaçao", "Arrowhead Stadium", "Kansas City"),
    (36, "2026-06-20", "F", "Tunisia", "Japan", "Estadio BBVA", "Guadalupe"),
    (37, "2026-06-21", "H", "Spain", "Saudi Arabia", "Mercedes-Benz Stadium", "Atlanta"),
    (38, "2026-06-21", "G", "Belgium", "Iran", "SoFi Stadium", "Inglewood"),
    (39, "2026-06-21", "H", "Uruguay", "Cape Verde", "Hard Rock Stadium", "Miami Gardens"),
    (40, "2026-06-21", "G", "New Zealand", "Egypt", "BC Place", "Vancouver"),
    (41, "2026-06-22", "J", "Argentina", "Austria", "AT&T Stadium", "Arlington"),
    (42, "2026-06-22", "I", "France", "Iraq", "Lincoln Financial Field", "Philadelphia"),
    (43, "2026-06-22", "I", "Norway", "Senegal", "MetLife Stadium", "East Rutherford"),
    (44, "2026-06-22", "J", "Jordan", "Algeria", "Levi's Stadium", "Santa Clara"),
    (45, "2026-06-23", "K", "Portugal", "Uzbekistan", "NRG Stadium", "Houston"),
    (46, "2026-06-23", "L", "England", "Ghana", "Gillette Stadium", "Foxborough"),
    (47, "2026-06-23", "L", "Panama", "Croatia", "BMO Field", "Toronto"),
    (48, "2026-06-23", "K", "Colombia", "DR Congo", "Estadio Akron", "Zapopan"),
    (49, "2026-06-24", "B", "Switzerland", "Canada", "BC Place", "Vancouver"),
    (50, "2026-06-24", "B", "Bosnia and Herzegovina", "Qatar", "Lumen Field", "Seattle"),
    (51, "2026-06-24", "C", "Scotland", "Brazil", "Hard Rock Stadium", "Miami Gardens"),
    (52, "2026-06-24", "C", "Morocco", "Haiti", "Mercedes-Benz Stadium", "Atlanta"),
    (53, "2026-06-24", "A", "Czechia", "Mexico", "Estadio Azteca", "Mexico City"),
    (54, "2026-06-24", "A", "South Africa", "South Korea", "Estadio BBVA", "Guadalupe"),
    (55, "2026-06-25", "E", "Ecuador", "Germany", "MetLife Stadium", "East Rutherford"),
    (56, "2026-06-25", "E", "Curaçao", "Ivory Coast", "Lincoln Financial Field", "Philadelphia"),
    (57, "2026-06-25", "F", "Japan", "Sweden", "AT&T Stadium", "Arlington"),
    (58, "2026-06-25", "F", "Tunisia", "Netherlands", "Arrowhead Stadium", "Kansas City"),
    (59, "2026-06-25", "D", "Türkiye", "United States", "SoFi Stadium", "Inglewood"),
    (60, "2026-06-25", "D", "Paraguay", "Australia", "Levi's Stadium", "Santa Clara"),
    (61, "2026-06-26", "I", "Norway", "France", "Gillette Stadium", "Foxborough"),
    (62, "2026-06-26", "I", "Senegal", "Iraq", "BMO Field", "Toronto"),
    (63, "2026-06-26", "H", "Cape Verde", "Saudi Arabia", "NRG Stadium", "Houston"),
    (64, "2026-06-26", "H", "Uruguay", "Spain", "Estadio Akron", "Zapopan"),
    (65, "2026-06-26", "G", "Egypt", "Iran", "Lumen Field", "Seattle"),
    (66, "2026-06-26", "G", "New Zealand", "Belgium", "BC Place", "Vancouver"),
    (67, "2026-06-27", "L", "Panama", "England", "MetLife Stadium", "East Rutherford"),
    (68, "2026-06-27", "L", "Croatia", "Ghana", "Lincoln Financial Field", "Philadelphia"),
    (69, "2026-06-27", "K", "Colombia", "Portugal", "Hard Rock Stadium", "Miami Gardens"),
    (70, "2026-06-27", "K", "DR Congo", "Uzbekistan", "Mercedes-Benz Stadium", "Atlanta"),
    (71, "2026-06-27", "J", "Algeria", "Austria", "Arrowhead Stadium", "Kansas City"),
    (72, "2026-06-27", "J", "Jordan", "Argentina", "AT&T Stadium", "Arlington"),
]

# Knockout matches: (id, date, stage, home_slot, away_slot, venue, city)
# Placeholders: "1X"=winner group X, "2X"=runner-up group X,
# "3rd-<groups>"=best third-placed from the listed group set, "WMnn"/"LMnn"=winner/loser of match nn.
K = [
    # Round of 32 (matches 73-88) - official bracket
    (73, "2026-06-28", "R32", "2A", "2B", "SoFi Stadium", "Inglewood"),
    (74, "2026-06-29", "R32", "1E", "3rd-ABCDF", "Gillette Stadium", "Foxborough"),
    (75, "2026-06-29", "R32", "1F", "2C", "Estadio BBVA", "Guadalupe"),
    (76, "2026-06-29", "R32", "1C", "2F", "NRG Stadium", "Houston"),
    (77, "2026-06-30", "R32", "1I", "3rd-CDFGH", "MetLife Stadium", "East Rutherford"),
    (78, "2026-06-30", "R32", "2E", "2I", "AT&T Stadium", "Arlington"),
    (79, "2026-06-30", "R32", "1A", "3rd-CEFHI", "Estadio Azteca", "Mexico City"),
    (80, "2026-07-01", "R32", "1L", "3rd-EHIJK", "Mercedes-Benz Stadium", "Atlanta"),
    (81, "2026-07-01", "R32", "1D", "3rd-BEFIJ", "Levi's Stadium", "Santa Clara"),
    (82, "2026-07-01", "R32", "1G", "3rd-AEHIJ", "Lumen Field", "Seattle"),
    (83, "2026-07-02", "R32", "2K", "2L", "BMO Field", "Toronto"),
    (84, "2026-07-02", "R32", "1H", "2J", "SoFi Stadium", "Inglewood"),
    (85, "2026-07-02", "R32", "1B", "3rd-EFGIJ", "BC Place", "Vancouver"),
    (86, "2026-07-03", "R32", "1J", "2H", "Hard Rock Stadium", "Miami Gardens"),
    (87, "2026-07-03", "R32", "1K", "3rd-DEIJL", "Arrowhead Stadium", "Kansas City"),
    (88, "2026-07-03", "R32", "2D", "2G", "AT&T Stadium", "Arlington"),
    # Round of 16 (89-96)
    (89, "2026-07-04", "R16", "WM74", "WM77", "Lincoln Financial Field", "Philadelphia"),
    (90, "2026-07-04", "R16", "WM73", "WM75", "NRG Stadium", "Houston"),
    (91, "2026-07-05", "R16", "WM76", "WM78", "MetLife Stadium", "East Rutherford"),
    (92, "2026-07-05", "R16", "WM79", "WM80", "Estadio Azteca", "Mexico City"),
    (93, "2026-07-06", "R16", "WM83", "WM84", "AT&T Stadium", "Arlington"),
    (94, "2026-07-06", "R16", "WM81", "WM82", "Lumen Field", "Seattle"),
    (95, "2026-07-07", "R16", "WM86", "WM88", "Mercedes-Benz Stadium", "Atlanta"),
    (96, "2026-07-07", "R16", "WM85", "WM87", "BC Place", "Vancouver"),
    # Quarterfinals (97-100)
    (97,  "2026-07-09", "QF", "WM89", "WM90", "Gillette Stadium", "Foxborough"),
    (98,  "2026-07-10", "QF", "WM93", "WM94", "SoFi Stadium", "Inglewood"),
    (99,  "2026-07-11", "QF", "WM91", "WM92", "Hard Rock Stadium", "Miami Gardens"),
    (100, "2026-07-11", "QF", "WM95", "WM96", "Arrowhead Stadium", "Kansas City"),
    # Semifinals (101-102)
    (101, "2026-07-14", "SF", "WM97", "WM98", "AT&T Stadium", "Arlington"),
    (102, "2026-07-15", "SF", "WM99", "WM100", "Mercedes-Benz Stadium", "Atlanta"),
    # Third place (103) and Final (104)
    (103, "2026-07-18", "third_place", "LM101", "LM102", "Hard Rock Stadium", "Miami Gardens"),
    (104, "2026-07-19", "final", "WM101", "WM102", "MetLife Stadium", "East Rutherford"),
]

frows = []
for mid, date, grp, home, away, venue, city in G:
    frows.append({
        "match_id": mid, "date": date, "stage": "group", "group": grp,
        "home_team": n(home), "away_team": n(away), "venue": venue, "city": city,
    })
for mid, date, stage, home, away, venue, city in K:
    frows.append({
        "match_id": mid, "date": date, "stage": stage, "group": "",
        "home_team": home, "away_team": away, "venue": venue, "city": city,
    })

fixtures_df = pd.DataFrame(frows, columns=[
    "match_id", "date", "stage", "group", "home_team", "away_team", "venue", "city"
])
fixtures_df = fixtures_df.sort_values("match_id").reset_index(drop=True)
fixtures_df.to_csv("data/fixtures.csv", index=False)

# === Verification ===
print("groups.csv rows:", len(groups_df))
print("group sizes:\n", groups_df.group.value_counts().sort_index())
print("fixtures.csv rows:", len(fixtures_df))
print("stage counts:\n", fixtures_df.stage.value_counts())
assert len(groups_df) == 48
assert (groups_df.group.value_counts() == 4).all()
assert len(fixtures_df) == 104
assert (fixtures_df.stage == "group").sum() == 72
assert list(fixtures_df.match_id) == list(range(1, 105))
# host check
print("hosts flagged:", sorted(groups_df[groups_df.is_host].team.unique()))
print("ALL ASSERTIONS PASSED")
