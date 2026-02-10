import argparse
import json
import sqlite3
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build query-friendly summary tables from payload_json in brand_* tables."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/db/franchise_top20_available.sqlite3",
        help="Path to sqlite db file",
    )
    return parser.parse_args()


def create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS summary_startup_cost")
    conn.execute("DROP TABLE IF EXISTS summary_franchise_fee")
    conn.execute("DROP TABLE IF EXISTS summary_deposit_info")

    conn.execute(
        """
        CREATE TABLE summary_startup_cost (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            brandMnno TEXT,
            brandNm TEXT,
            jnghdqrtrsMnno TEXT,
            source_year TEXT,
            yr TEXT,
            indutyLclasNm TEXT,
            indutyMlsfcNm TEXT,
            corpNm TEXT,
            jngBzmnJngAmt REAL,
            jngBzmnEduAmt REAL,
            jngBzmnAssrncAmt REAL,
            jngBzmnEtcAmt REAL,
            smtnAmt REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE summary_franchise_fee (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            brandMnno TEXT,
            brandNm TEXT,
            jnghdqrtrsMnno TEXT,
            source_year TEXT,
            jngBizCrtraYr TEXT,
            frstJnntSn INTEGER,
            jngAmtSeNm TEXT,
            jngAmtScopeVal TEXT,
            jngAmtGiveDdlnDateCn TEXT,
            jngAmtGvbkCndCn TEXT,
            jngAmtGvbkImprtyRsnCn TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE summary_deposit_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            brandMnno TEXT,
            brandNm TEXT,
            jnghdqrtrsMnno TEXT,
            source_year TEXT,
            jngBizCrtraYr TEXT,
            depoInstConmNm TEXT,
            depoChrgSpotDeptNm TEXT,
            depoInstAddr TEXT,
            depoInstTelno TEXT
        )
        """
    )

    conn.execute(
        "CREATE INDEX idx_summary_startup_brand_year ON summary_startup_cost(brandMnno, source_year)"
    )
    conn.execute(
        "CREATE INDEX idx_summary_fee_brand_year ON summary_franchise_fee(brandMnno, source_year)"
    )
    conn.execute(
        "CREATE INDEX idx_summary_depo_brand_year ON summary_deposit_info(brandMnno, source_year)"
    )


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def populate(conn: sqlite3.Connection) -> tuple[int, int, int]:
    brand_tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'brand_%' ORDER BY name"
        )
    ]
    startup_rows = 0
    fee_rows = 0
    depo_rows = 0

    for table in brand_tables:
        rows = conn.execute(
            f"""
            SELECT run_id, api_id, brandMnno, brandNm, jnghdqrtrsMnno, source_year, payload_json
            FROM {table}
            WHERE api_id IN ('15110265', '15125476', '15125482')
            """
        ).fetchall()

        for run_id, api_id, brand_mnno, brand_nm, hq_mnno, source_year, payload_json in rows:
            payload = json.loads(payload_json)
            if api_id == "15110265":
                conn.execute(
                    """
                    INSERT INTO summary_startup_cost (
                        run_id, brandMnno, brandNm, jnghdqrtrsMnno, source_year,
                        yr, indutyLclasNm, indutyMlsfcNm, corpNm,
                        jngBzmnJngAmt, jngBzmnEduAmt, jngBzmnAssrncAmt, jngBzmnEtcAmt, smtnAmt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        brand_mnno,
                        brand_nm,
                        hq_mnno,
                        source_year,
                        payload.get("yr"),
                        payload.get("indutyLclasNm"),
                        payload.get("indutyMlsfcNm"),
                        payload.get("corpNm"),
                        safe_float(payload.get("jngBzmnJngAmt")),
                        safe_float(payload.get("jngBzmnEduAmt")),
                        safe_float(payload.get("jngBzmnAssrncAmt")),
                        safe_float(payload.get("jngBzmnEtcAmt")),
                        safe_float(payload.get("smtnAmt")),
                    ),
                )
                startup_rows += 1
            elif api_id == "15125476":
                conn.execute(
                    """
                    INSERT INTO summary_franchise_fee (
                        run_id, brandMnno, brandNm, jnghdqrtrsMnno, source_year,
                        jngBizCrtraYr, frstJnntSn, jngAmtSeNm, jngAmtScopeVal,
                        jngAmtGiveDdlnDateCn, jngAmtGvbkCndCn, jngAmtGvbkImprtyRsnCn
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        brand_mnno,
                        brand_nm,
                        hq_mnno,
                        source_year,
                        payload.get("jngBizCrtraYr"),
                        payload.get("frstJnntSn"),
                        payload.get("jngAmtSeNm"),
                        payload.get("jngAmtScopeVal"),
                        payload.get("jngAmtGiveDdlnDateCn"),
                        payload.get("jngAmtGvbkCndCn"),
                        payload.get("jngAmtGvbkImprtyRsnCn"),
                    ),
                )
                fee_rows += 1
            elif api_id == "15125482":
                conn.execute(
                    """
                    INSERT INTO summary_deposit_info (
                        run_id, brandMnno, brandNm, jnghdqrtrsMnno, source_year,
                        jngBizCrtraYr, depoInstConmNm, depoChrgSpotDeptNm, depoInstAddr, depoInstTelno
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        brand_mnno,
                        brand_nm,
                        hq_mnno,
                        source_year,
                        payload.get("jngBizCrtraYr"),
                        payload.get("depoInstConmNm"),
                        payload.get("depoChrgSpotDeptNm"),
                        payload.get("depoInstAddr"),
                        payload.get("depoInstTelno"),
                    ),
                )
                depo_rows += 1

    conn.commit()
    return startup_rows, fee_rows, depo_rows


def main() -> int:
    args = parse_args()
    conn = sqlite3.connect(args.db_path)
    try:
        create_tables(conn)
        startup_rows, fee_rows, depo_rows = populate(conn)
        print(f"[INFO] summary_startup_cost rows={startup_rows}")
        print(f"[INFO] summary_franchise_fee rows={fee_rows}")
        print(f"[INFO] summary_deposit_info rows={depo_rows}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
