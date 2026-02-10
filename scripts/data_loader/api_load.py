import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_URL = "https://apis.data.go.kr/1130000/FftcBrandRlsInfo2_Service/getBrandinfo"


def load_api_key(cli_key: str | None) -> str:
    load_dotenv()

    candidates = [
        cli_key,
        os.getenv("DATA_GO_KR_SERVICE_KEY"),
        os.getenv("KFTC_API_KEY"),
    ]
    for key in candidates:
        if key and key.strip():
            return key.strip()

    raise ValueError(
        "API key not found. Pass --service-key or set DATA_GO_KR_SERVICE_KEY/KFTC_API_KEY."
    )


def request_page(
    service_key: str,
    year: int,
    page_no: int,
    num_of_rows: int,
    result_type: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    if result_type.lower() != "json":
        raise ValueError("Only JSON parsing is supported currently. Use --result-type json.")

    params = {
        "serviceKey": service_key,
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        "resultType": result_type,
        "jngBizCrtraYr": str(year),
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=20)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Network error after {max_retries} attempts (year={year}, page={page_no})."
                ) from exc
            time.sleep(2 ** (attempt - 1))
            continue

        if response.status_code in (401, 403):
            raise PermissionError(
                "Unauthorized (401/403). Check service key and API utilization approval status."
            )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Transient HTTP {response.status_code} after {max_retries} attempts "
                    f"(year={year}, page={page_no})."
                )
            time.sleep(2 ** (attempt - 1))
            continue

        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"JSON parse failed after {max_retries} attempts (year={year}, page={page_no})."
                ) from exc
            time.sleep(2 ** (attempt - 1))
            continue

        return payload

    raise RuntimeError(f"Unexpected retry loop exit (year={year}, page={page_no}).")


def normalize_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items", [])
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    if isinstance(items, dict):
        return [items]
    return []


def collect_year(
    service_key: str,
    year: int,
    num_of_rows: int,
    result_type: str,
    sleep_seconds: float,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    year_rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    try:
        first_payload = request_page(
            service_key=service_key,
            year=year,
            page_no=1,
            num_of_rows=num_of_rows,
            result_type=result_type,
        )
    except PermissionError:
        raise
    except Exception as exc:
        if fail_fast:
            raise
        print(f"[WARN] year={year} page=1 request failed: {exc}")
        return year_rows

    result_code = str(first_payload.get("resultCode", ""))
    if result_code and result_code != "00":
        msg = first_payload.get("resultMsg", "unknown")
        text = f"[WARN] year={year} resultCode={result_code}, resultMsg={msg}"
        if fail_fast:
            raise RuntimeError(text)
        print(text)

    total_count_raw = first_payload.get("totalCount", 0)
    try:
        total_count = int(total_count_raw)
    except (TypeError, ValueError):
        total_count = 0

    total_pages = math.ceil(total_count / num_of_rows) if total_count > 0 else 1

    first_items = normalize_items(first_payload)
    for item in first_items:
        row = dict(item)
        row["source_year"] = str(year)
        row["fetched_at"] = fetched_at
        year_rows.append(row)
    print(
        f"[INFO] year={year} page=1/{total_pages} +{len(first_items)} "
        f"(year_total={len(year_rows)}, total_count={total_count})"
    )

    for page_no in range(2, total_pages + 1):
        try:
            payload = request_page(
                service_key=service_key,
                year=year,
                page_no=page_no,
                num_of_rows=num_of_rows,
                result_type=result_type,
            )
            items = normalize_items(payload)
        except PermissionError:
            raise
        except Exception as exc:
            text = f"[WARN] year={year} page={page_no} failed: {exc}"
            if fail_fast:
                raise RuntimeError(text) from exc
            print(text)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        for item in items:
            row = dict(item)
            row["source_year"] = str(year)
            row["fetched_at"] = fetched_at
            year_rows.append(row)

        print(
            f"[INFO] year={year} page={page_no}/{total_pages} +{len(items)} "
            f"(year_total={len(year_rows)})"
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print(f"[INFO] year={year} finished rows={len(year_rows)}")
    return year_rows


def collect_year_range(
    start_year: int,
    end_year: int,
    service_key: str,
    num_of_rows: int,
    result_type: str,
    sleep_seconds: float,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year.")

    rows: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        print(f"[INFO] collecting year={year}")
        year_rows = collect_year(
            service_key=service_key,
            year=year,
            num_of_rows=num_of_rows,
            result_type=result_type,
            sleep_seconds=sleep_seconds,
            fail_fast=fail_fast,
        )
        rows.extend(year_rows)
        print(f"[INFO] cumulative rows={len(rows)}")
    return rows


def save_outputs(rows: list[dict[str, Any]], csv_path: str, json_path: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"[INFO] saved csv: {csv_path} (rows={len(df)})")
    print(f"[INFO] saved json: {json_path} (rows={len(rows)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load franchise brand list from data.go.kr OpenAPI."
    )
    parser.add_argument("--service-key", type=str, default=None, help="API service key")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--num-of-rows", type=int, default=100)
    parser.add_argument("--result-type", type=str, default="json", choices=["json", "xml"])
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/csv/franchise_brand_list_2016_2024.csv",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/csv/franchise_brand_list_2016_2024.json",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise immediately when page-level parse/request error happens.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        key = load_api_key(args.service_key)
        rows = collect_year_range(
            start_year=args.start_year,
            end_year=args.end_year,
            service_key=key,
            num_of_rows=args.num_of_rows,
            result_type=args.result_type,
            sleep_seconds=args.sleep_seconds,
            fail_fast=args.fail_fast,
        )
        save_outputs(rows, csv_path=args.csv_path, json_path=args.json_path)
        return 0
    except PermissionError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
