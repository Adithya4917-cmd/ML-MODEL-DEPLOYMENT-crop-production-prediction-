# Database part: SQLite (a single file, built into Python - nothing to install)
from pathlib import Path
from datetime import datetime
import sqlite3

DB_PATH = Path(__file__).resolve().parent.parent / "crop_app.db"


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row          # rows behave like dicts
    return con


def init_db():
    """create the tables the first time the app starts"""
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT,
                total_rows  INTEGER,
                ok_rows     INTEGER,
                created_at  TEXT
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                source                 TEXT,      -- 'form' or 'csv'
                upload_id              INTEGER,   -- which CSV it came from (NULL for form)
                country                TEXT,
                crop                   TEXT,
                year                   INTEGER,
                area_harvested_ha      REAL,
                temp_change_annual_c   REAL,
                temp_change_djf_c      REAL,
                temp_change_mam_c      REAL,
                temp_change_jja_c      REAL,
                temp_change_son_c      REAL,
                predicted_yield_t_ha   REAL,
                estimated_production_t REAL,
                created_at             TEXT
            )""")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_upload(filename, total_rows, ok_rows):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO uploads (filename, total_rows, ok_rows, created_at) VALUES (?, ?, ?, ?)",
            (filename, total_rows, ok_rows, now()))
        return cur.lastrowid


def add_predictions(rows, source, upload_id=None):
    """rows = list of dicts with the input columns + predicted_yield_t_ha + estimated_production_t"""
    with connect() as con:
        con.executemany("""
            INSERT INTO predictions (source, upload_id, country, crop, year, area_harvested_ha,
                temp_change_annual_c, temp_change_djf_c, temp_change_mam_c, temp_change_jja_c,
                temp_change_son_c, predicted_yield_t_ha, estimated_production_t, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(source, upload_id, r["country"], r["crop"], int(r["year"]), float(r["area_harvested_ha"]),
              float(r["temp_change_annual_c"]), float(r["temp_change_djf_c"]), float(r["temp_change_mam_c"]),
              float(r["temp_change_jja_c"]), float(r["temp_change_son_c"]),
              float(r["predicted_yield_t_ha"]), float(r["estimated_production_t"]), now())
             for r in rows])


def get_history(limit=50):
    with connect() as con:
        rows = con.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_uploads():
    with connect() as con:
        rows = con.execute("SELECT * FROM uploads ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]
