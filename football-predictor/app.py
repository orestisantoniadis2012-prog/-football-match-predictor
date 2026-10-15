"""
Football AI Predictor  (version 2)
==================================
Fixtures, results, standings and match predictions for Europe's top leagues.
Data comes from football-data.org (free plan, your token stays in secrets.toml).

HOW THE PREDICTIONS WORK (short version)
  1. We download every match of the season.
  2. For each team we measure attack strength and defence strength
     (recent matches count more, teams with few matches are pulled to the average).
  3. We turn that into "expected goals" for both teams.
  4. A Poisson model turns expected goals into the probability of every
     scoreline, and adding those up gives home win / draw / away win.
It is a statistical model, NOT a trained neural network, and it is not betting advice.

Run it with:   py -m streamlit run app.py
"""

import html
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Football AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# 1. SETTINGS  (safe to change)
# =====================================================================
API_BASE = "https://api.football-data.org/v4"
LOCAL_TZ = "Europe/Athens"   # all times are shown in this time zone
CACHE_SECONDS = 600          # keep downloaded data for 10 minutes (free plan = 10 requests / minute)
MIN_TEST = 20                # finished matches needed before the accuracy test appears
MAX_TEST = 150               # accuracy test uses at most the latest N matches

LEAGUES = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL": "Champions League",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "ELC": "Championship",
}

# How fast old matches fade away (days until a match counts only half)
MEMORY_DAYS = {"short": 75, "balanced": 150, "long": 300}

UPCOMING = ["SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"]
LIVE = ["IN_PLAY", "PAUSED"]
TABLE_STAGES = {"REGULAR_SEASON", "LEAGUE_STAGE", "GROUP_STAGE", ""}

C_HOME, C_DRAW, C_AWAY = "#3b82f6", "#8b95a5", "#f97316"
PICK_COLORS = [C_HOME, C_DRAW, C_AWAY]
FORM_EMOJI = {"W": "🟢", "D": "🟡", "L": "🔴"}

try:
    pd.Timestamp.now(tz=LOCAL_TZ)
except Exception:          # time zone database missing -> fall back to UTC
    LOCAL_TZ = "UTC"

# =====================================================================
# 2. TEXT  (English + Greek)
# =====================================================================
TEXT = {
    "en": {
        "subtitle": "Fixtures, results and match predictions for Europe's top leagues",
        "league": "League",
        "refresh": "🔄 Refresh data",
        "updated": "Data updated at {time}",
        "settings": "⚙️ Model settings",
        "use_prev": "Also learn from last season's matches",
        "prev_missing": "Last season's data isn't available on your plan (code {code}), so only this season is used.",
        "prev_skipped": "Couldn't load last season's data right now, so only this season is used.",
        "memory": "Model memory",
        "memory_help": "Short: recent matches matter most. Long: older matches count almost as much.",
        "mem_short": "Short",
        "mem_balanced": "Balanced",
        "mem_long": "Long",
        "loading": "Loading match data...",
        "tab_pred": "🔮 Predictions",
        "tab_matches": "📅 Matches",
        "tab_table": "🏆 Table",
        "tab_lab": "🧪 Lab",
        "tab_acc": "🎯 Accuracy",
        "footer": "Data from football-data.org. A learning project made for fun, not betting advice.",
        "tok_title": "API token not found",
        "tok_help1": "Create the file `.streamlit/secrets.toml` inside your project folder and put this line in it (with your own token):",
        "tok_help2": "Save it (Ctrl + S), stop the app (Ctrl + C in the terminal) and run `py -m streamlit run app.py` again.",
        "err_401": "The API token isn't valid (401). Check `.streamlit/secrets.toml`.",
        "err_403": "This league isn't included in your plan (403). Pick another league.",
        "err_404": "The API couldn't find this league (404).",
        "err_429": "Too many requests (429). Wait about a minute, then press Try again.",
        "err_other": "The API returned an error ({code}): {msg}",
        "err_net": "Couldn't reach football-data.org. Check your internet connection.",
        "retry": "Try again",
        "no_data": "There is no match data for this league yet.",
        "t_played": "Matches played",
        "t_goals": "Goals per match",
        "t_home": "Home wins",
        "t_draw": "Draws",
        "t_away": "Away wins",
        "t_over": "Over 2.5 goals",
        "days": "Show the next … days",
        "sort": "Sort by",
        "sort_date": "Date",
        "sort_conf": "Most confident first",
        "none_in_range": "No matches in this period, so here are the next fixtures instead.",
        "none_upcoming": "No upcoming matches found.",
        "md": "Matchday {n}",
        "pick": "Pick",
        "draw": "Draw",
        "predicted": "predicted score",
        "xg": "Expected goals",
        "over25": "Over 2.5",
        "btts": "Both teams score",
        "conf": "Confidence",
        "conf_high": "High",
        "conf_med": "Medium",
        "conf_low": "Low",
        "live": "LIVE",
        "model_info": "The model learned from {n} finished matches. Home advantage is ×{g}. Teams average {mu} goals per match.",
        "low_data": "There aren't many finished matches yet, so the predictions are rough.",
        "today": "Today",
        "no_today": "No matches today.",
        "delay_note": "On the free plan, scores can be delayed. Press Refresh data to check again.",
        "view": "Show",
        "v_up": "Upcoming",
        "v_res": "Results",
        "v_all": "All",
        "team": "Team",
        "all_teams": "All teams",
        "c_date": "Date",
        "c_md": "MD",
        "c_home": "Home",
        "c_away": "Away",
        "c_score": "Score",
        "c_status": "Status",
        "c_ph": "Home %",
        "c_pd": "Draw %",
        "c_pa": "Away %",
        "st_ft": "Full time",
        "st_live": "LIVE",
        "st_ht": "Half time",
        "st_sched": "Scheduled",
        "st_post": "Postponed",
        "st_susp": "Suspended",
        "st_canc": "Cancelled",
        "no_rows": "No matches found.",
        "table_note": "Calculated from this season's finished matches.",
        "c_pos": "#",
        "c_team": "Team",
        "c_p": "P",
        "c_w": "W",
        "c_dr": "D",
        "c_l": "L",
        "c_gf": "GF",
        "c_ga": "GA",
        "c_gd": "GD",
        "c_pts": "Pts",
        "c_form": "Form",
        "no_table": "No matches have been played yet this season.",
        "group": "Group {name}",
        "power_title": "Team strength ranking",
        "power_note": "Attack is how many goals a team is expected to score against an average team. Defence is how many it is expected to concede. Strength = attack minus defence.",
        "c_attack": "Attack",
        "c_defence": "Defence",
        "c_rating": "Strength",
        "lab_title": "Match lab: pick any two teams",
        "lab_home": "Home team",
        "lab_away": "Away team",
        "lab_same": "Pick two different teams.",
        "h2h": "Recent meetings",
        "no_h2h": "No meetings found in the loaded seasons.",
        "heat_title": "Scoreline probabilities (%)",
        "heat_hint": "Rows are {home} goals, columns are {away} goals. The orange outline is the predicted score.",
        "acc_title": "How good are the predictions?",
        "acc_intro": "We replay this season's finished matches. For each match day the model only uses games played before it, and then we compare its pick with what really happened.",
        "acc_need": "The test needs at least {n} finished matches this season.",
        "m_correct": "Correct picks",
        "m_base": "Always picking a home win",
        "m_loss": "Log loss (lower is better)",
        "m_tested": "Matches tested",
        "acc_note": "Football is very unpredictable. Even bookmakers and the best models pick the right result in roughly 50 to 55 percent of matches, and a random guess gets about 33 percent.",
        "by_conf": "Accuracy by confidence",
        "c_conf": "Confidence",
        "c_n": "Matches",
        "c_acc": "Correct",
        "recent": "Latest tested matches",
        "c_match": "Match",
        "c_pred": "Prediction",
        "c_actual": "Result",
        "how_title": "How does it work?",
        "how_body": (
            "**1. Collect results.** The app downloads every match of this season (and last season if your plan allows it).\n\n"
            "**2. Rate every team.** Each team gets an *attack* rating (goals it scores compared to an average team) and a *defence* rating (goals it concedes). "
            "Recent matches count more than old ones, and teams with few matches are pulled towards the average so one lucky game can't fool the model.\n\n"
            "**3. Expected goals.** For a fixture we combine the home team's attack with the away team's defence, add a home-advantage boost, and do the same for the away side.\n\n"
            "**4. Poisson maths.** Goals are rare random events, so the *Poisson distribution* turns expected goals into the probability of every scoreline (0-0, 1-0, 2-1 and so on). "
            "Adding up the scorelines gives home win, draw and away win.\n\n"
            "This is a statistical model, not a neural network. It doesn't know about injuries, line-ups, transfers or motivation, "
            "so use it for fun and learning, not for betting."
        ),
        "wdl": "WDL",
    },
    "el": {
        "subtitle": "Πρόγραμμα, αποτελέσματα και προβλέψεις αγώνων για τα κορυφαία ευρωπαϊκά πρωταθλήματα",
        "league": "Πρωτάθλημα",
        "refresh": "🔄 Ανανέωση δεδομένων",
        "updated": "Τελευταία ενημέρωση δεδομένων: {time}",
        "settings": "⚙️ Ρυθμίσεις μοντέλου",
        "use_prev": "Χρήση και των αγώνων της προηγούμενης σεζόν",
        "prev_missing": "Τα δεδομένα της προηγούμενης σεζόν δεν είναι διαθέσιμα στο πλάνο σου (κωδικός {code}), οπότε χρησιμοποιείται μόνο η τρέχουσα σεζόν.",
        "prev_skipped": "Δεν φορτώθηκαν τώρα τα δεδομένα της προηγούμενης σεζόν, οπότε χρησιμοποιείται μόνο η τρέχουσα σεζόν.",
        "memory": "Μνήμη μοντέλου",
        "memory_help": "Μικρή: μετράνε κυρίως οι πρόσφατοι αγώνες. Μεγάλη: και οι παλιότεροι αγώνες μετράνε σχεδόν το ίδιο.",
        "mem_short": "Μικρή",
        "mem_balanced": "Μεσαία",
        "mem_long": "Μεγάλη",
        "loading": "Φόρτωση δεδομένων αγώνων...",
        "tab_pred": "🔮 Προβλέψεις",
        "tab_matches": "📅 Αγώνες",
        "tab_table": "🏆 Βαθμολογία",
        "tab_lab": "🧪 Εργαστήριο",
        "tab_acc": "🎯 Ακρίβεια",
        "footer": "Δεδομένα από το football-data.org. Εκπαιδευτικό project για διασκέδαση, όχι συμβουλές στοιχηματισμού.",
        "tok_title": "Δεν βρέθηκε το API token",
        "tok_help1": "Φτιάξε το αρχείο `.streamlit/secrets.toml` μέσα στον φάκελο του project και γράψε αυτή τη γραμμή (με το δικό σου token):",
        "tok_help2": "Αποθήκευσέ το (Ctrl + S), σταμάτα την εφαρμογή (Ctrl + C στο terminal) και τρέξε ξανά `py -m streamlit run app.py`.",
        "err_401": "Το API token δεν είναι σωστό (401). Έλεγξε το `.streamlit/secrets.toml`.",
        "err_403": "Αυτό το πρωτάθλημα δεν περιλαμβάνεται στο πλάνο σου (403). Διάλεξε άλλο πρωτάθλημα.",
        "err_404": "Το API δεν βρήκε αυτό το πρωτάθλημα (404).",
        "err_429": "Πολλά αιτήματα (429). Περίμενε περίπου ένα λεπτό και μετά πάτα «Δοκίμασε ξανά».",
        "err_other": "Το API επέστρεψε σφάλμα ({code}): {msg}",
        "err_net": "Δεν μπόρεσα να συνδεθώ στο football-data.org. Έλεγξε τη σύνδεσή σου στο internet.",
        "retry": "Δοκίμασε ξανά",
        "no_data": "Δεν υπάρχουν ακόμα δεδομένα αγώνων για αυτό το πρωτάθλημα.",
        "t_played": "Αγώνες που έγιναν",
        "t_goals": "Γκολ ανά αγώνα",
        "t_home": "Νίκες γηπεδούχων",
        "t_draw": "Ισοπαλίες",
        "t_away": "Νίκες φιλοξενούμενων",
        "t_over": "Over 2.5 γκολ",
        "days": "Εμφάνιση για τις επόμενες … ημέρες",
        "sort": "Ταξινόμηση",
        "sort_date": "Ημερομηνία",
        "sort_conf": "Πρώτα οι πιο σίγουρες",
        "none_in_range": "Δεν υπάρχουν αγώνες σε αυτό το διάστημα, οπότε βλέπεις τους επόμενους αγώνες.",
        "none_upcoming": "Δεν βρέθηκαν επερχόμενοι αγώνες.",
        "md": "Αγωνιστική {n}",
        "pick": "Πρόβλεψη",
        "draw": "Ισοπαλία",
        "predicted": "προβλεπόμενο σκορ",
        "xg": "Αναμενόμενα γκολ",
        "over25": "Over 2.5",
        "btts": "Σκοράρουν και οι δύο",
        "conf": "Βεβαιότητα",
        "conf_high": "Υψηλή",
        "conf_med": "Μέτρια",
        "conf_low": "Χαμηλή",
        "live": "LIVE",
        "model_info": "Το μοντέλο έμαθε από {n} ολοκληρωμένους αγώνες. Το πλεονέκτημα έδρας είναι ×{g}. Οι ομάδες βάζουν κατά μέσο όρο {mu} γκολ ανά αγώνα.",
        "low_data": "Δεν έχουν παιχτεί ακόμα αρκετοί αγώνες, οπότε οι προβλέψεις είναι πρόχειρες.",
        "today": "Σήμερα",
        "no_today": "Δεν υπάρχουν αγώνες σήμερα.",
        "delay_note": "Στο δωρεάν πλάνο τα σκορ μπορεί να έχουν καθυστέρηση. Πάτα «Ανανέωση δεδομένων» για να ξαναελέγξεις.",
        "view": "Προβολή",
        "v_up": "Επερχόμενοι",
        "v_res": "Αποτελέσματα",
        "v_all": "Όλοι",
        "team": "Ομάδα",
        "all_teams": "Όλες οι ομάδες",
        "c_date": "Ημ/νία",
        "c_md": "Αγ.",
        "c_home": "Γηπεδούχος",
        "c_away": "Φιλοξενούμενος",
        "c_score": "Σκορ",
        "c_status": "Κατάσταση",
        "c_ph": "Γηπ. %",
        "c_pd": "Ισοπ. %",
        "c_pa": "Φιλ. %",
        "st_ft": "Τελικό",
        "st_live": "LIVE",
        "st_ht": "Ημίχρονο",
        "st_sched": "Προγραμματισμένος",
        "st_post": "Αναβλήθηκε",
        "st_susp": "Διακόπηκε",
        "st_canc": "Ακυρώθηκε",
        "no_rows": "Δεν βρέθηκαν αγώνες.",
        "table_note": "Υπολογίζεται από τους ολοκληρωμένους αγώνες της σεζόν.",
        "c_pos": "#",
        "c_team": "Ομάδα",
        "c_p": "Αγ",
        "c_w": "Ν",
        "c_dr": "Ι",
        "c_l": "Η",
        "c_gf": "ΓΥ",
        "c_ga": "ΓΚ",
        "c_gd": "ΔΤ",
        "c_pts": "Β",
        "c_form": "Φόρμα",
        "no_table": "Δεν έχουν παιχτεί ακόμα αγώνες αυτή τη σεζόν.",
        "group": "Όμιλος {name}",
        "power_title": "Κατάταξη δύναμης ομάδων",
        "power_note": "Επίθεση: τα γκολ που αναμένεται να βάλει μια ομάδα απέναντι σε μια μέση ομάδα. Άμυνα: τα γκολ που αναμένεται να δεχτεί. Δύναμη = επίθεση μείον άμυνα.",
        "c_attack": "Επίθεση",
        "c_defence": "Άμυνα",
        "c_rating": "Δύναμη",
        "lab_title": "Εργαστήριο αγώνων: διάλεξε οποιεσδήποτε δύο ομάδες",
        "lab_home": "Γηπεδούχος",
        "lab_away": "Φιλοξενούμενος",
        "lab_same": "Διάλεξε δύο διαφορετικές ομάδες.",
        "h2h": "Πρόσφατες αναμετρήσεις",
        "no_h2h": "Δεν βρέθηκαν αναμετρήσεις στις σεζόν που φορτώθηκαν.",
        "heat_title": "Πιθανότητες σκορ (%)",
        "heat_hint": "Οι γραμμές είναι τα γκολ του {home}, οι στήλες τα γκολ του {away}. Το πορτοκαλί πλαίσιο είναι το προβλεπόμενο σκορ.",
        "acc_title": "Πόσο καλές είναι οι προβλέψεις;",
        "acc_intro": "Ξαναπαίζουμε τους αγώνες της σεζόν που έχουν τελειώσει. Για κάθε ημέρα το μοντέλο χρησιμοποιεί μόνο αγώνες που έγιναν πριν, και μετά συγκρίνουμε την πρόβλεψή του με το πραγματικό αποτέλεσμα.",
        "acc_need": "Ο έλεγχος χρειάζεται τουλάχιστον {n} ολοκληρωμένους αγώνες αυτή τη σεζόν.",
        "m_correct": "Σωστές προβλέψεις",
        "m_base": "Αν έβαζες πάντα νίκη γηπεδούχου",
        "m_loss": "Log loss (χαμηλότερο = καλύτερο)",
        "m_tested": "Αγώνες στον έλεγχο",
        "acc_note": "Το ποδόσφαιρο είναι πολύ απρόβλεπτο. Ακόμα και οι bookmakers και τα καλύτερα μοντέλα βρίσκουν το σωστό αποτέλεσμα περίπου στο 50 με 55 τοις εκατό των αγώνων, ενώ μια τυχαία μαντεψιά πετυχαίνει περίπου 33 τοις εκατό.",
        "by_conf": "Ακρίβεια ανά βεβαιότητα",
        "c_conf": "Βεβαιότητα",
        "c_n": "Αγώνες",
        "c_acc": "Σωστές",
        "recent": "Τελευταίοι αγώνες στον έλεγχο",
        "c_match": "Αγώνας",
        "c_pred": "Πρόβλεψη",
        "c_actual": "Αποτέλεσμα",
        "how_title": "Πώς λειτουργεί;",
        "how_body": (
            "**1. Μαζεύουμε αποτελέσματα.** Η εφαρμογή κατεβάζει όλους τους αγώνες της σεζόν (και της προηγούμενης, αν το επιτρέπει το πλάνο σου).\n\n"
            "**2. Βαθμολογούμε κάθε ομάδα.** Κάθε ομάδα παίρνει βαθμό *επίθεσης* (πόσα γκολ βάζει σε σχέση με μια μέση ομάδα) και βαθμό *άμυνας* (πόσα γκολ δέχεται). "
            "Οι πρόσφατοι αγώνες μετράνε περισσότερο από τους παλιούς, και οι ομάδες με λίγους αγώνες τραβιούνται προς τον μέσο όρο, ώστε ένα τυχερό παιχνίδι να μην ξεγελάει το μοντέλο.\n\n"
            "**3. Αναμενόμενα γκολ.** Για κάθε αγώνα συνδυάζουμε την επίθεση του γηπεδούχου με την άμυνα του φιλοξενούμενου, προσθέτουμε το πλεονέκτημα έδρας και κάνουμε το ίδιο για τον φιλοξενούμενο.\n\n"
            "**4. Μαθηματικά Poisson.** Τα γκολ είναι σπάνια τυχαία γεγονότα, οπότε η *κατανομή Poisson* μετατρέπει τα αναμενόμενα γκολ σε πιθανότητα για κάθε σκορ (0-0, 1-0, 2-1 κ.λπ.). "
            "Αν προσθέσουμε τα σκορ, βγαίνουν οι πιθανότητες για νίκη γηπεδούχου, ισοπαλία και νίκη φιλοξενούμενου.\n\n"
            "Είναι στατιστικό μοντέλο, όχι νευρωνικό δίκτυο. Δεν ξέρει για τραυματισμούς, συνθέσεις, μεταγραφές ή κίνητρο, "
            "γι' αυτό χρησιμοποίησέ το για διασκέδαση και μάθηση, όχι για στοιχηματισμό."
        ),
        "wdl": "ΝΙΗ",
    },
}

LANG_LABELS = {"en": "EN", "el": "ΕΛ"}

WEEKDAYS = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "el": ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"],
}
MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "el": ["Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου", "Μαΐου", "Ιουνίου", "Ιουλίου",
           "Αυγούστου", "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου", "Δεκεμβρίου"],
}

STATUS_KEYS = {
    "FINISHED": "st_ft", "AWARDED": "st_ft", "IN_PLAY": "st_live", "PAUSED": "st_ht",
    "TIMED": "st_sched", "SCHEDULED": "st_sched", "POSTPONED": "st_post",
    "SUSPENDED": "st_susp", "CANCELLED": "st_canc",
}
DURATION_SUFFIX = {"EXTRA_TIME": " (aet)", "PENALTY_SHOOTOUT": " (pen)"}


def cur_lang():
    lang = st.session_state.get("lang", "en")
    return lang if lang in TEXT else "en"


def tr(key, **kw):
    """Look up a text in the chosen language."""
    text = TEXT[cur_lang()].get(key) or TEXT["en"].get(key, key)
    return text.format(**kw) if kw else text


def esc(value):
    """Make text safe to place inside HTML."""
    return html.escape("" if value is None else str(value), quote=True)


def fmt_day(ts):
    lang = cur_lang()
    return f"{WEEKDAYS[lang][ts.weekday()]}, {ts.day} {MONTHS[lang][ts.month - 1]}"


# =====================================================================
# 3. LOOK & FEEL  (CSS, works in light and dark mode, phone and PC)
# =====================================================================
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&display=swap');
:root{--mp-num:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif;--mp-line:rgba(128,128,128,.30);--mp-fill:rgba(128,128,128,.07)}
.block-container{max-width:1180px}
footer{visibility:hidden}
.mp-title{font-family:var(--mp-num);font-size:2.5rem;font-weight:700;line-height:1.05}
.mp-sub{opacity:.75;margin:2px 0 6px}
.mp-date{font-family:var(--mp-num);font-size:1.4rem;font-weight:600;margin:18px 0 6px}
.mp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px;margin:4px 0 8px}
.mp-card{border:1px solid var(--mp-line);border-left:5px solid var(--mp-line);border-radius:10px;padding:12px 14px;background:var(--mp-fill)}
.mp-top{display:flex;justify-content:space-between;gap:8px;font-size:.78rem;opacity:.8;margin-bottom:8px}
.mp-live{color:#ef4444;font-weight:700;opacity:1}
.mp-teams{display:grid;grid-template-columns:1fr auto 1fr;gap:6px;align-items:start}
.mp-team{text-align:center;min-width:0}
.mp-crest{width:40px;height:40px;object-fit:contain;display:block;margin:0 auto;font-size:1.6rem;line-height:40px;text-align:center}
.mp-name{font-weight:600;font-size:.9rem;line-height:1.15;margin-top:4px;overflow-wrap:anywhere}
.mp-mid{text-align:center;padding:0 4px;min-width:78px}
.mp-score{font-family:var(--mp-num);font-size:2.2rem;font-weight:700;line-height:1;white-space:nowrap;margin-top:6px}
.mp-cap{font-size:.68rem;opacity:.65;margin-top:2px}
.mp-form{display:flex;gap:3px;justify-content:center;margin-top:6px;min-height:17px}
.fb{display:inline-flex;align-items:center;justify-content:center;width:17px;height:17px;border-radius:4px;color:#fff;font-size:.66rem;font-weight:700}
.fb-w{background:#22a559}.fb-d{background:#8b95a5}.fb-l{background:#e5484d}
.mp-bar{display:flex;height:28px;border-radius:6px;overflow:hidden;margin-top:12px}
.mp-seg{display:flex;align-items:center;justify-content:center;color:#fff;font-family:var(--mp-num);font-size:1.05rem;font-weight:600;overflow:hidden;min-width:0}
.mp-legend{display:flex;justify-content:space-between;gap:6px;font-size:.72rem;opacity:.85;margin-top:5px}
.mp-legend span{display:inline-flex;align-items:center;gap:4px;min-width:0}
.mp-legend i{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.mp-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.mp-chip{font-size:.72rem;padding:2px 9px;border-radius:999px;border:1px solid var(--mp-line)}
.mp-chip.high{border-color:#22a559;color:#22a559}.mp-chip.med{border-color:#d9a400;color:#d9a400}.mp-chip.low{opacity:.8}
.mp-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:10px;margin:6px 0 10px}
.mp-tile{border:1px solid var(--mp-line);border-radius:8px;padding:8px 12px;background:var(--mp-fill)}
.mp-tile b{display:block;font-family:var(--mp-num);font-size:1.7rem;font-weight:700;line-height:1.1}
.mp-tile span{font-size:.74rem;opacity:.75}
table.hm{border-collapse:collapse;font-size:.76rem;text-align:center;width:100%;max-width:560px;margin:6px 0}
table.hm th,table.hm td{border:1px solid var(--mp-line);padding:5px 2px;min-width:34px}
table.hm th{opacity:.7;font-weight:600}
@media (max-width:600px){.mp-title{font-size:1.9rem}.mp-score{font-size:1.8rem}}
</style>"""


# =====================================================================
# 4. GETTING DATA FROM football-data.org
# =====================================================================
class ApiError(Exception):
    def __init__(self, status, message=""):
        super().__init__(f"{status} {message}")
        self.status = status
        self.message = message


def get_token():
    try:
        token = st.secrets["FOOTBALL_DATA_TOKEN"]
    except Exception:
        token = os.getenv("FOOTBALL_DATA_TOKEN", "")
    return str(token).strip().strip('"').strip("'").strip()


def api_get(path, params=None):
    response = requests.get(
        f"{API_BASE}{path}",
        headers={"X-Auth-Token": get_token()},
        params=params,
        timeout=20,
    )
    if response.status_code != 200:
        try:
            message = response.json().get("message", "")
        except ValueError:
            message = response.text[:200]
        raise ApiError(response.status_code, message)
    return response.json()


FRAME_COLUMNS = [
    "id", "utc", "matchday", "stage", "group", "status",
    "home_id", "home", "home_crest", "away_id", "away", "away_crest",
    "hg", "ag", "duration",
]


def regular_time_goals(score):
    """Goals after 90 minutes (extra time and penalties are ignored)."""
    for key in ("regularTime", "fullTime"):
        part = (score or {}).get(key) or {}
        if part.get("home") is not None and part.get("away") is not None:
            return part["home"], part["away"]
    return None, None


def matches_to_frame(raw_matches):
    """Turn the API's JSON into one tidy table (one row per match)."""
    rows = []
    for m in raw_matches:
        home, away = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        if home.get("id") is None or away.get("id") is None:
            continue  # knockout match where the teams are not known yet
        score = m.get("score") or {}
        hg, ag = regular_time_goals(score)
        rows.append({
            "id": m.get("id"),
            "utc": m.get("utcDate"),
            "matchday": m.get("matchday"),
            "stage": m.get("stage") or "",
            "group": m.get("group") or "",
            "status": m.get("status") or "",
            "home_id": home["id"],
            "home": home.get("shortName") or home.get("name") or "?",
            "home_crest": home.get("crest") or "",
            "away_id": away["id"],
            "away": away.get("shortName") or away.get("name") or "?",
            "away_crest": away.get("crest") or "",
            "hg": hg,
            "ag": ag,
            "duration": score.get("duration") or "REGULAR",
        })
    df = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    df["utc"] = pd.to_datetime(df["utc"], utc=True)
    for col in ("matchday", "hg", "ag"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["local"] = df["utc"].dt.tz_convert(LOCAL_TZ)
    df["finished"] = (df["status"] == "FINISHED") & df["hg"].notna()
    df["upcoming"] = df["status"].isin(UPCOMING)
    return df.sort_values("utc").reset_index(drop=True)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def load_season(code):
    """All matches of the current season (one API request)."""
    data = api_get(f"/competitions/{code}/matches")
    stamp = pd.Timestamp.now(tz=LOCAL_TZ).strftime("%H:%M")
    return matches_to_frame(data.get("matches", [])), stamp


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def load_previous_season(code, season):
    """Last season's matches. If your plan doesn't allow it we remember that, so we don't keep asking."""
    try:
        data = api_get(f"/competitions/{code}/matches", {"season": season})
        return matches_to_frame(data.get("matches", [])), None
    except ApiError as err:
        if err.status == 429:
            raise  # too many requests: don't remember this failure, just try again later
        return matches_to_frame([]), err.status


# =====================================================================
# 5. THE PREDICTION MODEL  (Poisson attack/defence model)
# =====================================================================
MAX_GOALS = 10
DC_RHO = -0.08  # small Dixon-Coles correction: 0-0 and 1-1 happen a bit more often than pure Poisson says
_K = np.arange(MAX_GOALS + 1)
_FACT = np.array([math.factorial(int(k)) for k in _K], dtype=float)


def poisson_pmf(lam):
    return np.exp(-lam) * lam ** _K / _FACT


class Model:
    """Team strengths learned from past matches."""

    def __init__(self, idx, att, dfn, gamma, mu, n_used):
        self.idx = idx        # team id -> position
        self.att = att        # attack strength (1.0 = average team)
        self.dfn = dfn        # goals conceded vs an average attacker (lower = better defence)
        self.gamma = gamma    # home advantage multiplier
        self.mu = mu          # average goals per team per match
        self.n_used = n_used  # number of matches the model learned from

    def expected_goals(self, home_id, away_id):
        i, j = self.idx.get(home_id), self.idx.get(away_id)
        att_h = self.att[i] if i is not None else 1.0
        dfn_h = self.dfn[i] if i is not None else self.mu
        att_a = self.att[j] if j is not None else 1.0
        dfn_a = self.dfn[j] if j is not None else self.mu
        lam_h = self.gamma * att_h * dfn_a
        lam_a = att_a * dfn_h
        return float(np.clip(lam_h, 0.15, 5.0)), float(np.clip(lam_a, 0.15, 5.0))


def fit_model(finished, team_ids, ref_time, half_life, prior=4.0, iters=60):
    """
    Learn attack/defence strengths from matches played before `ref_time`.
    Expected goals:  home = gamma * attack(home) * defence(away),  away = attack(away) * defence(home)
    """
    idx = {t: k for k, t in enumerate(team_ids)}
    n = len(idx)
    d = finished[finished["utc"] < ref_time]
    if len(d) < 4:  # almost no data: everybody is average
        return Model(idx, np.ones(n), np.full(n, 1.35), 1.25, 1.35, len(d))

    h = d["home_id"].map(idx).to_numpy().astype(int)
    a = d["away_id"].map(idx).to_numpy().astype(int)
    gh = d["hg"].to_numpy(dtype=float)
    ga = d["ag"].to_numpy(dtype=float)
    age_days = (ref_time - d["utc"]).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (age_days / half_life)  # recent matches weigh more

    mu = float((w * (gh + ga)).sum() / (2 * w.sum()))
    att, dfn, gamma = np.ones(n), np.full(n, mu), 1.25
    c = prior  # "pretend" every team also played `c` average matches (keeps small samples sensible)

    for _ in range(iters):
        num = np.bincount(h, w * gh, n) + np.bincount(a, w * ga, n) + c * mu
        den = np.bincount(h, w * gamma * dfn[a], n) + np.bincount(a, w * dfn[h], n) + c * mu
        att = num / den
        num = np.bincount(a, w * gh, n) + np.bincount(h, w * ga, n) + c * mu
        den = np.bincount(a, w * gamma * att[h], n) + np.bincount(h, w * att[a], n) + c
        dfn = num / den
        gamma = float((w * gh).sum() / (w * att[h] * dfn[a]).sum())
        scale = att.mean()
        att, dfn = att / scale, dfn * scale
    return Model(idx, att, dfn, gamma, mu, len(d))


def score_matrix(lam_h, lam_a, rho=DC_RHO):
    """Probability of every scoreline: matrix[i][j] = P(home scores i, away scores j)."""
    m = np.outer(poisson_pmf(lam_h), poisson_pmf(lam_a))
    m[0, 0] *= 1 - lam_h * lam_a * rho
    m[0, 1] *= 1 + lam_h * rho
    m[1, 0] *= 1 + lam_a * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def round_probs(p):
    """Round three probabilities to whole percents that add up to exactly 100."""
    raw = np.asarray(p, dtype=float) * 100
    base = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - base))[: 100 - int(base.sum())]:
        base[k] += 1
    return [int(x) for x in base]


def conf_level(top_prob):
    if top_prob >= 0.60:
        return "high"
    if top_prob >= 0.47:
        return "med"
    return "low"


def predict_match(model, home_id, away_id):
    lam_h, lam_a = model.expected_goals(home_id, away_id)
    m = score_matrix(lam_h, lam_a)
    ii, jj = np.indices(m.shape)
    probs = np.array([m[ii > jj].sum(), m[ii == jj].sum(), m[ii < jj].sum()])
    pick = int(probs.argmax())  # 0 = home win, 1 = draw, 2 = away win
    allowed = [ii > jj, ii == jj, ii < jj][pick]
    best = np.unravel_index(np.where(allowed, m, -1.0).argmax(), m.shape)
    return {
        "m": m,
        "lam_h": lam_h,
        "lam_a": lam_a,
        "probs": probs,
        "pct": round_probs(probs),
        "pick": pick,
        "score": (int(best[0]), int(best[1])),  # most likely score that fits the pick
        "over25": int(round(m[ii + jj >= 3].sum() * 100)),
        "btts": int(round(m[1:, 1:].sum() * 100)),
        "top": float(probs.max()),
    }


def run_backtest(finished_all, cur_finished, team_ids, half_life):
    """Replay finished matches: predict each one using only the games played before that day."""
    test = cur_finished.sort_values("utc").tail(MAX_TEST)
    rows = []
    for day, games in test.groupby(test["utc"].dt.floor("D")):
        model = fit_model(finished_all, team_ids, day, half_life)
        for r in games.itertuples():
            p = predict_match(model, r.home_id, r.away_id)
            actual = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            rows.append({
                "date": r.local,
                "home": r.home,
                "away": r.away,
                "score": f"{int(r.hg)} – {int(r.ag)}",
                "pick": p["pick"],
                "pred_score": f"{p['score'][0]} – {p['score'][1]}",
                "actual": actual,
                "correct": p["pick"] == actual,
                "loss": float(-math.log(max(p["probs"][actual], 1e-9))),
                "conf": conf_level(p["top"]),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def cached_backtest(_finished_all, _cur_finished, _team_ids, key, half_life):
    # `key` changes whenever new results arrive; the underscore arguments are not hashed.
    return run_backtest(_finished_all, _cur_finished, _team_ids, half_life)


# =====================================================================
# 6. TEAM FORM AND STANDINGS
# =====================================================================
def to_long(fin):
    """One row per team per match (so we can add up results for each team)."""
    home = pd.DataFrame({"team_id": fin["home_id"], "team": fin["home"],
                         "gf": fin["hg"], "ga": fin["ag"], "utc": fin["utc"]})
    away = pd.DataFrame({"team_id": fin["away_id"], "team": fin["away"],
                         "gf": fin["ag"], "ga": fin["hg"], "utc": fin["utc"]})
    long = pd.concat([home, away], ignore_index=True).sort_values("utc", kind="stable")
    long["res"] = np.where(long["gf"] > long["ga"], "W", np.where(long["gf"] == long["ga"], "D", "L"))
    return long


def build_form_map(fin):
    """team id -> list with the results of the last 5 matches, oldest first."""
    if fin.empty:
        return {}
    long = to_long(fin)
    return long.groupby("team_id")["res"].apply(lambda s: list(s.tail(5))).to_dict()


def standings(part, forms):
    """League table for one group of matches (lists every team, even before they have played)."""
    teams = pd.concat([
        part[["home_id", "home"]].rename(columns={"home_id": "team_id", "home": "team"}),
        part[["away_id", "away"]].rename(columns={"away_id": "team_id", "away": "team"}),
    ]).drop_duplicates("team_id")
    fin = part[part["finished"]]
    if fin.empty:
        stats = pd.DataFrame(columns=["team_id", "p", "w", "d", "l", "gf", "ga"])
    else:
        long = to_long(fin)
        long["w"] = (long["res"] == "W").astype(int)
        long["d"] = (long["res"] == "D").astype(int)
        long["l"] = (long["res"] == "L").astype(int)
        stats = long.groupby("team_id").agg(
            p=("res", "size"), w=("w", "sum"), d=("d", "sum"), l=("l", "sum"),
            gf=("gf", "sum"), ga=("ga", "sum"),
        ).reset_index()
    t = teams.merge(stats, on="team_id", how="left").fillna(0)
    for col in ("p", "w", "d", "l", "gf", "ga"):
        t[col] = t[col].astype(int)
    t["gd"] = t["gf"] - t["ga"]
    t["pts"] = 3 * t["w"] + t["d"]
    t = t.sort_values(["pts", "gd", "gf", "team"], ascending=[False, False, False, True]).reset_index(drop=True)
    form = ["".join(FORM_EMOJI[r] for r in forms.get(tid, [])) for tid in t["team_id"]]
    return pd.DataFrame({
        tr("c_pos"): range(1, len(t) + 1),
        tr("c_team"): t["team"],
        tr("c_p"): t["p"], tr("c_w"): t["w"], tr("c_dr"): t["d"], tr("c_l"): t["l"],
        tr("c_gf"): t["gf"], tr("c_ga"): t["ga"], tr("c_gd"): t["gd"], tr("c_pts"): t["pts"],
        tr("c_form"): form,
    })


# =====================================================================
# 7. HTML PIECES (cards, bars, tiles, heat-map)
# =====================================================================
def form_html(results):
    letters = dict(zip("WDL", tr("wdl")))
    return "".join(f'<span class="fb fb-{r.lower()}">{letters[r]}</span>' for r in results)


def team_block(name, crest, results):
    if crest:
        img = f'<img class="mp-crest" src="{esc(crest)}" alt="" loading="lazy">'
    else:
        img = '<div class="mp-crest">⚽</div>'
    return (f'<div class="mp-team">{img}<div class="mp-name">{esc(name)}</div>'
            f'<div class="mp-form">{form_html(results)}</div></div>')


def prob_bar(pct):
    segs = []
    for value, color in zip(pct, PICK_COLORS):
        label = f"{value}%" if value >= 12 else ""
        segs.append(f'<div class="mp-seg" style="width:{value}%;background:{color}">{label}</div>')
    return '<div class="mp-bar">' + "".join(segs) + "</div>"


def legend_html(home, away):
    items = ((C_HOME, home), (C_DRAW, tr("draw")), (C_AWAY, away))
    spans = "".join(f'<span><i style="background:{col}"></i>{esc(name)}</span>' for col, name in items)
    return f'<div class="mp-legend">{spans}</div>'


def match_card(info, p, forms):
    home, away = info["home"], info["away"]
    level = conf_level(p["top"])
    pick_name = [home, tr("draw"), away][p["pick"]]
    chips = (
        f'<span class="mp-chip"><b>{esc(tr("pick"))}:</b> {esc(pick_name)}</span>'
        f'<span class="mp-chip">{esc(tr("xg"))} {p["lam_h"]:.1f} – {p["lam_a"]:.1f}</span>'
        f'<span class="mp-chip">{esc(tr("over25"))}: {p["over25"]}%</span>'
        f'<span class="mp-chip">{esc(tr("btts"))}: {p["btts"]}%</span>'
        f'<span class="mp-chip {level}">{esc(tr("conf"))}: {esc(tr("conf_" + level))}</span>'
    )
    live_class = ' class="mp-live"' if info["live"] else ""
    return (
        f'<div class="mp-card" style="border-left-color:{PICK_COLORS[p["pick"]]}">'
        f'<div class="mp-top"><span>{esc(info["left"])}</span><span{live_class}>{esc(info["right"])}</span></div>'
        '<div class="mp-teams">'
        f'{team_block(home, info["hcrest"], forms.get(info["hid"], []))}'
        f'<div class="mp-mid"><div class="mp-score">{p["score"][0]}–{p["score"][1]}</div>'
        f'<div class="mp-cap">{esc(tr("predicted"))}</div></div>'
        f'{team_block(away, info["acrest"], forms.get(info["aid"], []))}'
        "</div>"
        f'{prob_bar(p["pct"])}{legend_html(home, away)}'
        f'<div class="mp-chips">{chips}</div>'
        "</div>"
    )


def card_info(r, show_date):
    """Collect the texts a card needs from one match row."""
    if r.stage in TABLE_STAGES:
        left = tr("md", n=int(r.matchday)) if pd.notna(r.matchday) else ""
    else:
        left = r.stage.replace("_", " ").title()
    live = r.status in LIVE
    if live:
        right = f"● {tr('live')}" + (f" {int(r.hg)}–{int(r.ag)}" if pd.notna(r.hg) else "")
    else:
        right = r.local.strftime("%d/%m %H:%M" if show_date else "%H:%M")
    return {"home": r.home, "away": r.away, "hcrest": r.home_crest, "acrest": r.away_crest,
            "hid": r.home_id, "aid": r.away_id, "left": left, "right": right, "live": live}


def render_cards(items, forms, show_date):
    cards = "".join(match_card(card_info(r, show_date), p, forms) for r, p in items)
    st.markdown(f'<div class="mp-grid">{cards}</div>', unsafe_allow_html=True)


def tiles_html(items):
    tiles = "".join(f'<div class="mp-tile"><b>{esc(v)}</b><span>{esc(label)}</span></div>' for v, label in items)
    return f'<div class="mp-tiles">{tiles}</div>'


def heatmap_html(m, best, n=6):
    """Table of scoreline probabilities; darker blue = more likely."""
    top = m[:n, :n].max()
    head = "<tr><th></th>" + "".join(f"<th>{j}</th>" for j in range(n)) + "</tr>"
    body = []
    for i in range(n):
        cells = []
        for j in range(n):
            alpha = 0.06 + 0.84 * m[i, j] / top
            style = f"background:rgba(59,130,246,{alpha:.2f});"
            if alpha > 0.55:
                style += "color:#fff;"
            if (i, j) == best:
                style += "outline:2px solid #f97316;outline-offset:-2px;font-weight:700;"
            cells.append(f'<td style="{style}">{m[i, j] * 100:.1f}</td>')
        body.append(f"<tr><th>{i}</th>{''.join(cells)}</tr>")
    return f'<table class="hm">{head}{"".join(body)}</table>'


def show_table(df, height=None):
    kwargs = {"hide_index": True}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


# =====================================================================
# 8. THE FIVE TABS
# =====================================================================
def tab_predictions(c):
    col1, col2 = st.columns(2)
    days = col1.slider(tr("days"), 1, 30, 7, key="days")
    sort_by = col2.radio(tr("sort"), ["date", "conf"], horizontal=True,
                         format_func=lambda x: tr("sort_" + x), key="sort_by")

    upcoming = c.allm[c.allm["upcoming"] & (c.allm["utc"] >= c.now - pd.Timedelta(hours=12))]
    if upcoming.empty:
        st.info(tr("none_upcoming"))
        return
    shown = upcoming[upcoming["utc"] <= c.now + pd.Timedelta(days=days)]
    if shown.empty:
        st.info(tr("none_in_range"))
        shown = upcoming.head(6)

    items = [(r, predict_match(c.model, r.home_id, r.away_id)) for r in shown.itertuples()]
    if c.model.n_used < 30:
        st.warning(tr("low_data"))

    if sort_by == "conf":
        items.sort(key=lambda it: it[1]["top"], reverse=True)
        render_cards(items, c.form_map, show_date=True)
    else:
        groups = {}
        for r, p in items:
            groups.setdefault(r.local.strftime("%Y-%m-%d"), []).append((r, p))
        for rows in groups.values():
            st.markdown(f'<div class="mp-date">{esc(fmt_day(rows[0][0].local))}</div>', unsafe_allow_html=True)
            render_cards(rows, c.form_map, show_date=False)

    st.caption(tr("model_info", n=c.model.n_used, g=f"{c.model.gamma:.2f}", mu=f"{c.model.mu:.2f}"))


def match_table(df, c):
    rows = []
    for r in df.itertuples():
        score = "–"
        if pd.notna(r.hg):
            score = f"{int(r.hg)} – {int(r.ag)}" + DURATION_SUFFIX.get(r.duration, "")
        probs = ["", "", ""]
        if r.upcoming:
            probs = [f"{x}%" for x in predict_match(c.model, r.home_id, r.away_id)["pct"]]
        rows.append({
            tr("c_date"): r.local.strftime("%d/%m %H:%M"),
            tr("c_md"): int(r.matchday) if pd.notna(r.matchday) else "",
            tr("c_home"): r.home,
            tr("c_score"): score,
            tr("c_away"): r.away,
            tr("c_status"): tr(STATUS_KEYS.get(r.status, "st_sched")),
            tr("c_ph"): probs[0], tr("c_pd"): probs[1], tr("c_pa"): probs[2],
        })
    return pd.DataFrame(rows)


def tab_matches(c):
    st.subheader(tr("today"))
    today = pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d")
    todays = c.cur[c.cur["local"].dt.strftime("%Y-%m-%d") == today]
    if todays.empty:
        st.caption(tr("no_today"))
    else:
        show_table(match_table(todays, c))
        st.caption(tr("delay_note"))

    col1, col2 = st.columns(2)
    view = col1.radio(tr("view"), ["up", "res", "all"], horizontal=True,
                      format_func=lambda x: tr("v_" + x), key="view")
    ids = sorted(c.names, key=lambda t: c.names[t].lower())
    team = col2.selectbox(tr("team"), [None] + ids, key=f"team_filter_{c.code}",
                          format_func=lambda t: tr("all_teams") if t is None else c.names[t])

    df = c.cur
    if team is not None:
        df = df[(df["home_id"] == team) | (df["away_id"] == team)]
    if view == "up":
        df = df[df["upcoming"]].sort_values("utc")
    elif view == "res":
        df = df[df["finished"]].sort_values("utc", ascending=False)
    if df.empty:
        st.info(tr("no_rows"))
    else:
        show_table(match_table(df, c), height=min(38 * (len(df) + 1) + 3, 760))


def tab_table(c):
    st.caption(tr("table_note"))
    df = c.cur[c.cur["stage"].isin(TABLE_STAGES)]
    if df.empty or not df["finished"].any():
        st.info(tr("no_table"))
        return
    groups = sorted(df["group"].unique())
    for g in groups:
        table = standings(df[df["group"] == g], c.form_map)
        if len(groups) > 1 and g:
            nice = g.replace("GROUP_", "").replace("_", " ").title()
            st.markdown(f"**{tr('group', name=nice)}**")
        show_table(table, height=min(38 * (len(table) + 1) + 3, 900))


def power_table(c):
    ids = list(c.names)
    ix = [c.model.idx[t] for t in ids]
    att, dfn = c.model.att[ix], c.model.dfn[ix]
    scored = att * dfn.mean()      # goals scored against an average team
    conceded = dfn * att.mean()    # goals conceded against an average team
    df = pd.DataFrame({
        tr("c_team"): [c.names[t] for t in ids],
        tr("c_attack"): scored.round(2),
        tr("c_defence"): conceded.round(2),
        tr("c_rating"): (scored - conceded).round(2),
    }).sort_values(tr("c_rating"), ascending=False).reset_index(drop=True)
    df.insert(0, tr("c_pos"), range(1, len(df) + 1))
    return df


def tab_lab(c):
    st.subheader(tr("power_title"))
    st.caption(tr("power_note"))
    show_table(power_table(c), height=min(38 * (len(c.names) + 1) + 3, 760))

    st.subheader(tr("lab_title"))
    ids = sorted(c.names, key=lambda t: c.names[t].lower())
    d_home, d_away = 0, min(1, len(ids) - 1)
    if c.next_match is not None:
        d_home, d_away = ids.index(c.next_match.home_id), ids.index(c.next_match.away_id)
    col1, col2 = st.columns(2)
    home_id = col1.selectbox(tr("lab_home"), ids, index=d_home, format_func=c.names.get, key=f"lab_home_{c.code}")
    away_id = col2.selectbox(tr("lab_away"), ids, index=d_away, format_func=c.names.get, key=f"lab_away_{c.code}")
    if home_id == away_id:
        st.warning(tr("lab_same"))
        return

    p = predict_match(c.model, home_id, away_id)
    info = {"home": c.names[home_id], "away": c.names[away_id],
            "hcrest": c.crests.get(home_id, ""), "acrest": c.crests.get(away_id, ""),
            "hid": home_id, "aid": away_id, "left": "", "right": "", "live": False}
    st.markdown(f'<div class="mp-grid">{match_card(info, p, c.form_map)}</div>', unsafe_allow_html=True)

    st.markdown(f"**{tr('heat_title')}**")
    st.caption(tr("heat_hint", home=info["home"], away=info["away"]))
    st.markdown(heatmap_html(p["m"], p["score"]), unsafe_allow_html=True)

    st.markdown(f"**{tr('h2h')}**")
    pair = c.fin_all[((c.fin_all["home_id"] == home_id) & (c.fin_all["away_id"] == away_id))
                     | ((c.fin_all["home_id"] == away_id) & (c.fin_all["away_id"] == home_id))]
    pair = pair.sort_values("utc", ascending=False).head(5)
    if pair.empty:
        st.caption(tr("no_h2h"))
    else:
        show_table(pd.DataFrame({
            tr("c_date"): pair["local"].dt.strftime("%d/%m/%Y"),
            tr("c_home"): pair["home"],
            tr("c_score"): [f"{int(a)} – {int(b)}" for a, b in zip(pair["hg"], pair["ag"])],
            tr("c_away"): pair["away"],
        }))


def tab_accuracy(c):
    st.subheader(tr("acc_title"))
    st.write(tr("acc_intro"))
    if len(c.cur_fin) < MIN_TEST:
        st.info(tr("acc_need", n=MIN_TEST))
    else:
        key = f"{c.code}|{len(c.fin_all)}|{c.fin_all['utc'].max()}"
        bt = cached_backtest(c.fin_all, c.cur_fin, c.team_ids, key, c.half_life)
        base = float((bt["actual"] == 0).mean())
        st.markdown(tiles_html([
            (f"{bt['correct'].mean() * 100:.0f}%", tr("m_correct")),
            (f"{base * 100:.0f}%", tr("m_base")),
            (f"{bt['loss'].mean():.3f}", tr("m_loss")),
            (str(len(bt)), tr("m_tested")),
        ]), unsafe_allow_html=True)
        st.caption(tr("acc_note"))

        st.markdown(f"**{tr('by_conf')}**")
        order = ["high", "med", "low"]
        rows = []
        for level in order:
            part = bt[bt["conf"] == level]
            if len(part):
                rows.append({tr("c_conf"): tr("conf_" + level), tr("c_n"): len(part),
                             tr("c_acc"): f"{part['correct'].mean() * 100:.0f}%"})
        show_table(pd.DataFrame(rows))

        st.markdown(f"**{tr('recent')}**")
        last = bt.sort_values("date", ascending=False).head(15)
        labels = []
        for r in last.itertuples():
            labels.append([r.home, tr("draw"), r.away][r.pick])
        show_table(pd.DataFrame({
            tr("c_date"): last["date"].dt.strftime("%d/%m"),
            tr("c_match"): [f"{h} – {a}" for h, a in zip(last["home"], last["away"])],
            tr("c_pred"): [f"{lab} ({ps})" for lab, ps in zip(labels, last["pred_score"])],
            tr("c_actual"): last["score"],
            "": ["✅" if ok else "❌" for ok in last["correct"]],
        }))

    with st.expander(tr("how_title")):
        st.markdown(tr("how_body"))


# =====================================================================
# 9. PUTTING IT ALL TOGETHER
# =====================================================================
def show_api_error(err):
    messages = {401: "err_401", 403: "err_403", 404: "err_404", 429: "err_429"}
    if err.status in messages:
        st.error(tr(messages[err.status]))
    else:
        st.error(tr("err_other", code=err.status, msg=err.message))


def team_directory(df):
    both = pd.concat([
        df[["home_id", "home", "home_crest"]].set_axis(["id", "name", "crest"], axis=1),
        df[["away_id", "away", "away_crest"]].set_axis(["id", "name", "crest"], axis=1),
    ]).drop_duplicates("id").set_index("id")
    return both["name"].to_dict(), both["crest"].to_dict()


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    # ---- header + language switch
    head_left, head_right = st.columns([5, 1])
    with head_left:
        st.markdown(
            f'<div class="mp-title">⚽ Football AI Predictor</div>'
            f'<div class="mp-sub">{esc(tr("subtitle"))}</div>',
            unsafe_allow_html=True,
        )
    with head_right:
        st.radio("Language", list(LANG_LABELS), key="lang", horizontal=True,
                 format_func=LANG_LABELS.get, label_visibility="collapsed")

    # ---- token check
    if not get_token():
        st.error(f"**{tr('tok_title')}**")
        st.markdown(tr("tok_help1"))
        st.code('FOOTBALL_DATA_TOKEN = "PASTE_YOUR_TOKEN_HERE"', language="toml")
        st.markdown(tr("tok_help2"))
        st.stop()

    # ---- controls
    code = st.selectbox(tr("league"), list(LEAGUES), format_func=LEAGUES.get, key="league")
    btn_col, stamp_col = st.columns([1, 2])
    btn_col.button(tr("refresh"), on_click=st.cache_data.clear)
    stamp = stamp_col.empty()
    with st.expander(tr("settings")):
        use_prev = st.checkbox(tr("use_prev"), value=True, key="use_prev")
        memory = st.select_slider(tr("memory"), options=list(MEMORY_DAYS), value="balanced",
                                  format_func=lambda k: tr("mem_" + k), help=tr("memory_help"), key="memory")
    half_life = MEMORY_DAYS[memory]

    # ---- load data
    try:
        with st.spinner(tr("loading")):
            cur, fetched_at = load_season(code)
    except ApiError as err:
        show_api_error(err)
        st.button(tr("retry"))
        st.stop()
    except requests.exceptions.RequestException:
        st.error(tr("err_net"))
        st.button(tr("retry"))
        st.stop()
    if cur.empty:
        st.info(tr("no_data"))
        st.stop()
    stamp.caption(tr("updated", time=fetched_at))

    prev = matches_to_frame([])
    if use_prev:
        try:
            prev, prev_error = load_previous_season(code, int(cur["utc"].min().year) - 1)
            if prev_error:
                st.caption(tr("prev_missing", code=prev_error))
        except (ApiError, requests.exceptions.RequestException):
            st.caption(tr("prev_skipped"))

    allm = pd.concat([prev, cur], ignore_index=True) if len(prev) else cur
    allm = allm.drop_duplicates("id").sort_values("utc").reset_index(drop=True)

    # ---- prepare everything the tabs need
    now = pd.Timestamp.now(tz="UTC")
    fin_all = allm[allm["finished"]]
    cur_fin = cur[cur["finished"]]
    team_ids = list(pd.unique(pd.concat([allm["home_id"], allm["away_id"]])))
    names, crests = team_directory(cur)
    next_up = cur[cur["upcoming"] & (cur["utc"] >= now - pd.Timedelta(hours=12))]
    c = SimpleNamespace(
        code=code, cur=cur, allm=allm, fin_all=fin_all, cur_fin=cur_fin, team_ids=team_ids,
        names=names, crests=crests, now=now, half_life=half_life,
        model=fit_model(fin_all, team_ids, now, half_life),
        form_map=build_form_map(fin_all),
        next_match=next(next_up.itertuples()) if len(next_up) else None,
    )

    # ---- league overview
    if len(cur_fin):
        goals = cur_fin["hg"] + cur_fin["ag"]
        st.markdown(tiles_html([
            (str(len(cur_fin)), tr("t_played")),
            (f"{goals.mean():.2f}", tr("t_goals")),
            (f"{(cur_fin['hg'] > cur_fin['ag']).mean() * 100:.0f}%", tr("t_home")),
            (f"{(cur_fin['hg'] == cur_fin['ag']).mean() * 100:.0f}%", tr("t_draw")),
            (f"{(cur_fin['hg'] < cur_fin['ag']).mean() * 100:.0f}%", tr("t_away")),
            (f"{(goals >= 3).mean() * 100:.0f}%", tr("t_over")),
        ]), unsafe_allow_html=True)

    # ---- tabs
    tabs = st.tabs([tr("tab_pred"), tr("tab_matches"), tr("tab_table"), tr("tab_lab"), tr("tab_acc")])
    with tabs[0]:
        tab_predictions(c)
    with tabs[1]:
        tab_matches(c)
    with tabs[2]:
        tab_table(c)
    with tabs[3]:
        tab_lab(c)
    with tabs[4]:
        tab_accuracy(c)

    st.caption(tr("footer"))


main()