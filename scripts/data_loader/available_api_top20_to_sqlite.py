import argparse
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class ApiSpec:
    api_id: str
    api_name: str
    url: str
    year_param: str
    filter_mode: str  # by_brand, by_hq, global_brand, global_hq
    id_param: str | None = None


API_SPECS: list[ApiSpec] = [
    ApiSpec(
        api_id="15125467",
        api_name="브랜드 목록",
        url="https://apis.data.go.kr/1130000/FftcBrandRlsInfo2_Service/getBrandinfo",
        year_param="jngBizCrtraYr",
        filter_mode="global_brand",
    ),
    ApiSpec(
        api_id="15125441",
        api_name="가맹본부 등록 목록",
        url="https://apis.data.go.kr/1130000/FftcJnghdqrtrsRgsInfo2_Service/getjnghdqrtrsListinfo",
        year_param="jngBizCrtraYr",
        filter_mode="global_hq",
    ),
    ApiSpec(
        api_id="15125456",
        api_name="가맹본부 재무정보",
        url="https://apis.data.go.kr/1130000/FftcjnghdqrtrsFnnrInfo2_Service/getjnghdqrtrsFnlttinfo",
        year_param="jngBizCrtraYr",
        filter_mode="by_hq",
        id_param="jnghdqrtrsMnno",
    ),
    ApiSpec(
        api_id="15110241",
        api_name="브랜드별 가맹점 현황",
        url="https://apis.data.go.kr/1130000/FftcBrandFrcsStatsService/getBrandFrcsStats",
        year_param="yr",
        filter_mode="global_brand",
    ),
    ApiSpec(
        api_id="15110265",
        api_name="브랜드별 창업 금액",
        url="https://apis.data.go.kr/1130000/FftcBrandFntnStatsService/getBrandFntnStats",
        year_param="yr",
        filter_mode="global_brand",
    ),
    ApiSpec(
        api_id="15125476",
        api_name="브랜드 가맹금 정보",
        url="https://apis.data.go.kr/1130000/FftcbrandfrcsjnntinfoService/getbrandFrcsJnntinfo",
        year_param="jngBizCrtraYr",
        filter_mode="by_brand",
        id_param="brandMnno",
    ),
    ApiSpec(
        api_id="15125482",
        api_name="브랜드 가맹금 예치",
        url="https://apis.data.go.kr/1130000/FftcbrandjnntdepoinstinfoService/getbrandJnntDepoInstinfo",
        year_param="jngBizCrtraYr",
        filter_mode="by_brand",
        id_param="brandMnno",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect accessible APIs for top20 brands and store by-brand tables in SQLite."
    )
    parser.add_argument("--brand-csv", type=str, default="data/csv/franchise_brand_top20_famous.csv")
    parser.add_argument("--db-path", type=str, default="data/db/franchise_top20_available.sqlite3")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--num-of-rows", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--service-key", type=str, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_api_key(cli_key: str | None) -> str:
    load_dotenv()
    for key in (cli_key, os.getenv("DATA_GO_KR_SERVICE_KEY"), os.getenv("KFTC_API_KEY")):
        if key and key.strip():
            return key.strip()
    raise ValueError(
        "API key not found. Set --service-key or DATA_GO_KR_SERVICE_KEY or KFTC_API_KEY."
    )


def norm_table_name(brand_mnno: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in brand_mnno)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"brand_{safe.strip('_')}"


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def create_meta_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_run_log (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            status TEXT NOT NULL,
            message TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_call_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            api_id TEXT NOT NULL,
            year TEXT,
            brandMnno TEXT,
            page_no INTEGER,
            http_status INTEGER,
            result_code TEXT,
            result_msg TEXT,
            row_count INTEGER,
            error TEXT
        )
        """
    )
    conn.commit()


def create_brand_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            api_id TEXT NOT NULL,
            brandMnno TEXT NOT NULL,
            brandNm TEXT,
            jnghdqrtrsMnno TEXT,
            source_year TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_api_year ON {table_name}(api_id, source_year)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_brand_api ON {table_name}(brandMnno, api_id)"
    )


def request_with_retry(
    url: str,
    params: dict[str, str],
    max_retries: int = 3,
) -> tuple[int, dict[str, Any] | None, str | None]:
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=25)
        except requests.RequestException as exc:
            if attempt == max_retries:
                return -1, None, str(exc)
            time.sleep(2 ** (attempt - 1))
            continue

        if resp.status_code in (429,) or 500 <= resp.status_code < 600:
            if attempt == max_retries:
                return resp.status_code, None, resp.text[:300]
            time.sleep(2 ** (attempt - 1))
            continue

        if resp.status_code in (401, 403):
            return resp.status_code, None, resp.text[:300]

        try:
            payload = resp.json()
        except ValueError:
            return resp.status_code, None, "JSON parse failed"

        return resp.status_code, payload, None

    return -1, None, "unreachable"


def normalize_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    items = payload.get("items", [])
    if isinstance(items, list):
        return [it for it in items if isinstance(it, dict)]
    if isinstance(items, dict):
        return [items]
    return []


def insert_call_log(
    conn: sqlite3.Connection,
    run_id: str,
    api_id: str,
    year: int | None,
    brand_mnno: str | None,
    page_no: int | None,
    http_status: int | None,
    result_code: str | None,
    result_msg: str | None,
    row_count: int,
    error: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO api_call_log (
            run_id, api_id, year, brandMnno, page_no, http_status, result_code, result_msg, row_count, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            api_id,
            str(year) if year is not None else None,
            brand_mnno,
            page_no,
            http_status,
            result_code,
            result_msg,
            row_count,
            error,
        ),
    )


def insert_brand_row(
    conn: sqlite3.Connection,
    table_name: str,
    run_id: str,
    api_id: str,
    brand_mnno: str,
    brand_nm: str,
    hq_mnno: str,
    source_year: int,
    fetched_at: str,
    payload: dict[str, Any],
) -> None:
    conn.execute(
        f"""
        INSERT INTO {table_name} (
            run_id, api_id, brandMnno, brandNm, jnghdqrtrsMnno, source_year, fetched_at, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            api_id,
            brand_mnno,
            brand_nm,
            hq_mnno,
            str(source_year),
            fetched_at,
            json.dumps(payload, ensure_ascii=False),
        ),
    )


def full_refresh_delete(
    conn: sqlite3.Connection,
    table_name: str,
    api_ids: list[str],
    start_year: int,
    end_year: int,
) -> None:
    placeholders = ",".join("?" for _ in api_ids)
    conn.execute(
        f"""
        DELETE FROM {table_name}
        WHERE api_id IN ({placeholders})
          AND CAST(source_year AS INTEGER) BETWEEN ? AND ?
        """,
        (*api_ids, start_year, end_year),
    )


def build_top20_maps(df: pd.DataFrame) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], set[str], set[str]]:
    brands = {}
    hqs = {}
    brand_set = set()
    hq_set = set()
    for _, row in df.iterrows():
        brand_mnno = str(row.get("brandMnno", "")).strip()
        brand_nm = str(row.get("brandNm", "")).strip()
        hq_mnno = str(row.get("jnghdqrtrsMnno", "")).strip()
        if not brand_mnno:
            continue
        brands[brand_mnno] = {
            "brandMnno": brand_mnno,
            "brandNm": brand_nm,
            "jnghdqrtrsMnno": hq_mnno,
        }
        brand_set.add(brand_mnno)
        if hq_mnno:
            hq_set.add(hq_mnno)
            if hq_mnno not in hqs:
                hqs[hq_mnno] = {
                    "brandMnno": brand_mnno,
                    "brandNm": brand_nm,
                    "jnghdqrtrsMnno": hq_mnno,
                }
    return brands, hqs, brand_set, hq_set


def normalize_brand_name(name: str) -> str:
    return "".join(str(name).strip().lower().split())


def build_brand_name_maps(
    by_brand: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    raw_name_map: dict[str, dict[str, str]] = {}
    norm_name_map: dict[str, dict[str, str]] = {}
    for info in by_brand.values():
        brand_nm = str(info.get("brandNm", "")).strip()
        if not brand_nm:
            continue
        raw_name_map.setdefault(brand_nm, info)
        norm_name_map.setdefault(normalize_brand_name(brand_nm), info)
    return raw_name_map, norm_name_map


def resolve_target(
    item: dict[str, Any],
    mode: str,
    by_brand: dict[str, dict[str, str]],
    by_hq: dict[str, dict[str, str]],
    brand_set: set[str],
    hq_set: set[str],
    brand_name_raw_map: dict[str, dict[str, str]],
    brand_name_norm_map: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    item_brand = str(item.get("brandMnno", "")).strip()
    item_hq = str(item.get("jnghdqrtrsMnno", "")).strip()
    item_brand_nm = str(item.get("brandNm", "")).strip()
    if mode in {"by_brand", "global_brand"}:
        if item_brand and item_brand in brand_set:
            return by_brand[item_brand]
        # Some APIs do not provide brandMnno; fallback to brandNm matching.
        if item_brand_nm and item_brand_nm in brand_name_raw_map:
            return brand_name_raw_map[item_brand_nm]
        norm = normalize_brand_name(item_brand_nm)
        if norm and norm in brand_name_norm_map:
            return brand_name_norm_map[norm]
        return None
    if mode in {"by_hq", "global_hq"}:
        if item_hq and item_hq in hq_set:
            return by_hq[item_hq]
        return None
    return None


def collect_for_api(
    conn: sqlite3.Connection,
    run_id: str,
    spec: ApiSpec,
    service_key: str,
    start_year: int,
    end_year: int,
    num_of_rows: int,
    sleep_seconds: float,
    by_brand: dict[str, dict[str, str]],
    by_hq: dict[str, dict[str, str]],
    brand_set: set[str],
    hq_set: set[str],
    brand_name_raw_map: dict[str, dict[str, str]],
    brand_name_norm_map: dict[str, dict[str, str]],
    fail_fast: bool,
) -> None:
    fetched_at = datetime.now(timezone.utc).isoformat()
    for year in range(start_year, end_year + 1):
        print(f"[INFO] api={spec.api_id} year={year} start")
        base_params = {
            "serviceKey": service_key,
            "pageNo": "1",
            "numOfRows": str(num_of_rows),
            "resultType": "json",
            spec.year_param: str(year),
        }
        if spec.filter_mode in {"by_brand", "by_hq"}:
            ids = by_brand.keys() if spec.filter_mode == "by_brand" else by_hq.keys()
            total_ids = len(by_brand) if spec.filter_mode == "by_brand" else len(by_hq)
            done_ids = 0
            for filter_id in ids:
                params = dict(base_params)
                params[spec.id_param or ""] = filter_id
                status, payload, error = request_with_retry(spec.url, params=params)
                if error is not None:
                    insert_call_log(
                        conn=conn,
                        run_id=run_id,
                        api_id=spec.api_id,
                        year=year,
                        brand_mnno=filter_id if spec.filter_mode == "by_brand" else None,
                        page_no=1,
                        http_status=status if status >= 0 else None,
                        result_code=None,
                        result_msg=None,
                        row_count=0,
                        error=error,
                    )
                    conn.commit()
                    if fail_fast:
                        raise RuntimeError(f"{spec.api_id} year={year} id={filter_id} failed: {error}")
                    done_ids += 1
                    print(
                        f"[WARN] api={spec.api_id} year={year} id_progress={done_ids}/{total_ids} "
                        f"status={status if status >= 0 else 'ERR'}"
                    )
                    continue

                items = normalize_items(payload)
                result_code = str((payload or {}).get("resultCode", ""))
                result_msg = str((payload or {}).get("resultMsg", ""))
                saved = 0
                for item in items:
                    target = resolve_target(
                        item=item,
                        mode=spec.filter_mode,
                        by_brand=by_brand,
                        by_hq=by_hq,
                        brand_set=brand_set,
                        hq_set=hq_set,
                        brand_name_raw_map=brand_name_raw_map,
                        brand_name_norm_map=brand_name_norm_map,
                    )
                    if not target:
                        continue
                    table_name = norm_table_name(target["brandMnno"])
                    insert_brand_row(
                        conn=conn,
                        table_name=table_name,
                        run_id=run_id,
                        api_id=spec.api_id,
                        brand_mnno=target["brandMnno"],
                        brand_nm=target["brandNm"],
                        hq_mnno=target["jnghdqrtrsMnno"],
                        source_year=year,
                        fetched_at=fetched_at,
                        payload=item,
                    )
                    saved += 1
                insert_call_log(
                    conn=conn,
                    run_id=run_id,
                    api_id=spec.api_id,
                    year=year,
                    brand_mnno=filter_id if spec.filter_mode == "by_brand" else None,
                    page_no=1,
                    http_status=status,
                    result_code=result_code,
                    result_msg=result_msg,
                    row_count=saved,
                    error=None,
                )
                conn.commit()
                done_ids += 1
                print(
                    f"[INFO] api={spec.api_id} year={year} id_progress={done_ids}/{total_ids} "
                    f"saved={saved} status={status} resultCode={result_code or '-'}"
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        else:
            status, first_payload, error = request_with_retry(spec.url, params=base_params)
            if error is not None:
                insert_call_log(
                    conn=conn,
                    run_id=run_id,
                    api_id=spec.api_id,
                    year=year,
                    brand_mnno=None,
                    page_no=1,
                    http_status=status if status >= 0 else None,
                    result_code=None,
                    result_msg=None,
                    row_count=0,
                    error=error,
                )
                conn.commit()
                if fail_fast:
                    raise RuntimeError(f"{spec.api_id} year={year} page=1 failed: {error}")
                continue

            total_count = int((first_payload or {}).get("totalCount", 0) or 0)
            total_pages = math.ceil(total_count / num_of_rows) if total_count > 0 else 1
            print(
                f"[INFO] api={spec.api_id} year={year} totalCount={total_count} totalPages={total_pages}"
            )
            for page_no in range(1, total_pages + 1):
                payload = first_payload if page_no == 1 else None
                status_cur = status if page_no == 1 else None
                if page_no > 1:
                    page_params = dict(base_params)
                    page_params["pageNo"] = str(page_no)
                    status_cur, payload, error = request_with_retry(spec.url, params=page_params)
                    if error is not None:
                        insert_call_log(
                            conn=conn,
                            run_id=run_id,
                            api_id=spec.api_id,
                            year=year,
                            brand_mnno=None,
                            page_no=page_no,
                            http_status=status_cur if status_cur and status_cur >= 0 else None,
                            result_code=None,
                            result_msg=None,
                            row_count=0,
                            error=error,
                        )
                        conn.commit()
                        if fail_fast:
                            raise RuntimeError(
                                f"{spec.api_id} year={year} page={page_no} failed: {error}"
                            )
                        print(
                            f"[WARN] api={spec.api_id} year={year} page={page_no}/{total_pages} "
                            f"status={status_cur if status_cur is not None else 'ERR'}"
                        )
                        continue

                items = normalize_items(payload)
                result_code = str((payload or {}).get("resultCode", ""))
                result_msg = str((payload or {}).get("resultMsg", ""))
                saved = 0
                for item in items:
                    target = resolve_target(
                        item=item,
                        mode=spec.filter_mode,
                        by_brand=by_brand,
                        by_hq=by_hq,
                        brand_set=brand_set,
                        hq_set=hq_set,
                        brand_name_raw_map=brand_name_raw_map,
                        brand_name_norm_map=brand_name_norm_map,
                    )
                    if not target:
                        continue
                    table_name = norm_table_name(target["brandMnno"])
                    insert_brand_row(
                        conn=conn,
                        table_name=table_name,
                        run_id=run_id,
                        api_id=spec.api_id,
                        brand_mnno=target["brandMnno"],
                        brand_nm=target["brandNm"],
                        hq_mnno=target["jnghdqrtrsMnno"],
                        source_year=year,
                        fetched_at=fetched_at,
                        payload=item,
                    )
                    saved += 1
                insert_call_log(
                    conn=conn,
                    run_id=run_id,
                    api_id=spec.api_id,
                    year=year,
                    brand_mnno=None,
                    page_no=page_no,
                    http_status=status_cur,
                    result_code=result_code,
                    result_msg=result_msg,
                    row_count=saved,
                    error=None,
                )
                conn.commit()
                print(
                    f"[INFO] api={spec.api_id} year={year} page={page_no}/{total_pages} "
                    f"saved={saved} status={status_cur} resultCode={result_code or '-'}"
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        print(f"[INFO] api={spec.api_id} year={year} done")


def main() -> int:
    args = parse_args()
    if args.start_year > args.end_year:
        print("[ERROR] start-year must be <= end-year")
        return 1

    try:
        service_key = load_api_key(args.service_key)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    df = pd.read_csv(args.brand_csv)
    by_brand, by_hq, brand_set, hq_set = build_top20_maps(df)
    brand_name_raw_map, brand_name_norm_map = build_brand_name_maps(by_brand)
    if not by_brand:
        print("[ERROR] No brand rows found from brand csv.")
        return 1

    conn = connect_db(args.db_path)
    try:
        create_meta_tables(conn)
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO api_run_log (run_id, started_at, status, message) VALUES (?, ?, ?, ?)",
            (run_id, started_at, "running", None),
        )

        api_ids = [s.api_id for s in API_SPECS]
        for info in by_brand.values():
            table_name = norm_table_name(info["brandMnno"])
            create_brand_table(conn, table_name)
            full_refresh_delete(
                conn=conn,
                table_name=table_name,
                api_ids=api_ids,
                start_year=args.start_year,
                end_year=args.end_year,
            )
        conn.commit()

        for spec in API_SPECS:
            print(f"[INFO] collecting api={spec.api_id} ({spec.api_name})")
            collect_for_api(
                conn=conn,
                run_id=run_id,
                spec=spec,
                service_key=service_key,
                start_year=args.start_year,
                end_year=args.end_year,
                num_of_rows=args.num_of_rows,
                sleep_seconds=args.sleep_seconds,
                by_brand=by_brand,
                by_hq=by_hq,
                brand_set=brand_set,
                hq_set=hq_set,
                brand_name_raw_map=brand_name_raw_map,
                brand_name_norm_map=brand_name_norm_map,
                fail_fast=args.fail_fast,
            )

        ended_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE api_run_log SET ended_at=?, status=?, message=? WHERE run_id=?",
            (ended_at, "success", "completed", run_id),
        )
        conn.commit()
        print(f"[INFO] done run_id={run_id}, db={args.db_path}")
        return 0
    except Exception as exc:
        ended_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE api_run_log SET ended_at=?, status=?, message=? WHERE run_id=(SELECT run_id FROM api_run_log ORDER BY rowid DESC LIMIT 1)",
            (ended_at, "failed", str(exc)[:1000]),
        )
        conn.commit()
        print(f"[ERROR] {exc}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
