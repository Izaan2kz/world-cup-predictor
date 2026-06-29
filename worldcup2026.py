"""World Cup 2026 teams, groups, and data constants."""

import os
import requests

NAME_MAP = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "Iran": "IR Iran",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Turkey": "Türkiye",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "Curacao": "Curaçao",
}

FLAG_EMOJI = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "Korea Republic": "🇰🇷", "Czechia": "🇨🇿",
    "Canada": "🇨🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "USA": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Türkiye": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Côte d'Ivoire": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Tunisia": "🇹🇳", "Sweden": "🇸🇪",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "IR Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cabo Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Iraq": "🇮🇶", "Senegal": "🇸🇳", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Congo DR": "🇨🇩", "Portugal": "🇵🇹", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}

GROUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Qatar", "Switzerland", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Iraq", "Senegal", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Congo DR", "Portugal", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_TEAMS = [t for g in GROUPS.values() for t in g]
TEAM_TO_GROUP = {t: g for g, teams in GROUPS.items() for t in teams}

R32_SLOTS = [
    (73, "2A", "2B", None),
    (74, "1E", None, ["A", "B", "C", "D", "F"]),
    (75, "1F", "2C", None),
    (76, "1C", "2F", None),
    (77, "1I", None, ["C", "D", "F", "G", "H"]),
    (78, "2E", "2I", None),
    (79, "1A", None, ["C", "E", "F", "H", "I"]),
    (80, "1L", None, ["E", "H", "I", "J", "K"]),
    (81, "1D", None, ["B", "E", "F", "I", "J"]),
    (82, "1G", None, ["A", "E", "H", "I", "J"]),
    (83, "2K", "2L", None),
    (84, "1H", "2J", None),
    (85, "1B", None, ["E", "F", "G", "I", "J"]),
    (86, "1J", "2H", None),
    (87, "1K", None, ["D", "E", "I", "J", "L"]),
    (88, "2D", "2G", None),
]

R16_PAIRS = [(73, 74), (75, 76), (77, 78), (79, 80), (81, 82), (83, 84), (85, 86), (87, 88)]

STAGES = ["group_exit", "r32", "r16", "quarters", "semis", "final", "champion"]


def download_data(force=False):
    from datetime import datetime
    DATA_FILES = ["results.csv"]
    BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/"
    for fname in DATA_FILES:
        if force or not os.path.exists(fname):
            print(f"Downloading {fname}...")
            r = requests.get(BASE_URL + fname)
            r.raise_for_status()
            with open(fname, "wb") as f:
                f.write(r.content)
            print(f"Saved {fname}")
    mod_time = datetime.fromtimestamp(os.path.getmtime("results.csv"))
    print(f"Data last updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
