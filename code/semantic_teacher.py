#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from activity_taxonomy import (
    ACTIVITY12_CATEGORIES,
    ACTIVITY12_DESCRIPTIONS,
    activity12_mapping_sha256,
    fine_to_activity12,
    validate_activity12_coverage,
)


TEACHER_MODEL = "deepseek-v4-flash"
CACHE_SCHEMA = "activity12_full_distribution"
SCENARIO_PREVIEW_LENGTH = 3


class LLMClient:
    model_name = TEACHER_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        max_retries: int = 2,
        retry_backoff: float = 2.0,
        timeout: float = 30.0,
        pool_timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        use_env_proxy: bool = True,
    ):
        self.api_key = (api_key or os.environ.get("LLM_API_KEY", "")).strip()
        self.base_url = str(base_url)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.timeout = float(timeout)
        self.pool_timeout = float(pool_timeout)
        self.max_connections = max(1, int(max_connections))
        self.max_keepalive_connections = min(
            self.max_connections,
            max(0, int(max_keepalive_connections)),
        )
        self.use_env_proxy = bool(use_env_proxy)
        self._client = None
        self._client_lock = threading.Lock()

    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("provide --api_key or set LLM_API_KEY")
        if "sk-xxxx" in self.api_key.lower() or "api_key" in self.api_key.lower():
            raise ValueError("placeholder API key detected")
        try:
            self.api_key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("API key must contain ASCII characters only") from exc

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import httpx
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("install openai and httpx to generate the teacher cache") from exc
            self._validate_api_key()
            timeout = httpx.Timeout(self.timeout, pool=self.pool_timeout)
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout,
                http_client=httpx.Client(
                    timeout=timeout,
                    limits=limits,
                    trust_env=self.use_env_proxy,
                ),
            )
        return self._client

    def _request(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ):
        return self._get_client().chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=max(64, int(max_tokens)),
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )

    @staticmethod
    def _parse_response(response) -> dict:
        if not getattr(response, "choices", None):
            raise ValueError("teacher response has no choices")
        content = (getattr(response.choices[0].message, "content", "") or "").strip()
        if not content:
            raise ValueError("teacher response is empty")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("teacher response must be a JSON object")
        return parsed

    def query(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.05,
        max_tokens: int = 1024,
    ) -> dict:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(prompt, system_prompt, temperature, max_tokens)
                return self._parse_response(response)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_backoff ** attempt + random.uniform(0.0, 0.5)
                time.sleep(delay)
        raise last_error


def _coerce_timezone_minutes(value) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _parse_utc_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(str(value), "%a %b %d %H:%M:%S +0000 %Y")
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except (TypeError, ValueError):
            return None


def _format_utc_offset(offset_minutes: int) -> str:
    sign = "+" if offset_minutes >= 0 else "-"
    absolute = abs(int(offset_minutes))
    return f"UTC{sign}{absolute // 60:02d}:{absolute % 60:02d}"


def _parse_utc_time(value: str, timezone_minutes=None) -> dict:
    utc_datetime = _parse_utc_datetime(value)
    if utc_datetime is None:
        return {
            "weekend": False,
            "hour": 0,
            "time_str": str(value),
            "utc_datetime": None,
        }
    offset_minutes = _coerce_timezone_minutes(timezone_minutes)
    local_datetime = utc_datetime + timedelta(minutes=offset_minutes)
    hour = local_datetime.hour
    if hour < 5:
        period = "late night"
    elif hour < 9:
        period = "morning"
    elif hour < 12:
        period = "late morning"
    elif hour < 14:
        period = "midday"
    elif hour < 18:
        period = "afternoon"
    elif hour < 22:
        period = "evening"
    else:
        period = "late evening"
    weekday = local_datetime.strftime("%A")
    return {
        "weekend": weekday in {"Saturday", "Sunday"},
        "hour": hour,
        "time_str": (
            f"{local_datetime.strftime('%Y-%m-%d %H:%M')} {weekday} "
            f"({period}, {_format_utc_offset(offset_minutes)})"
        ),
        "utc_datetime": utc_datetime,
    }


def _row_timezone_minutes(row) -> int:
    return _coerce_timezone_minutes(row.get("timezone") if hasattr(row, "get") else None)


def _format_elapsed(start: datetime | None, end: datetime | None) -> str | None:
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    if seconds < 0:
        raise ValueError("trajectory timestamps are not chronological")
    hours, minutes = divmod(int(round(seconds / 60.0)), 60)
    if hours and minutes:
        return f"{hours}h{minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _haversine_km(first, second) -> float | None:
    try:
        lat1, lon1 = float(first["latitude"]), float(first["longitude"])
        lat2, lon2 = float(second["latitude"]), float(second["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (lat1, lon1, lat2, lon2)):
        return None
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    latitude_delta = math.radians(lat2 - lat1)
    longitude_delta = math.radians(lon2 - lon1)
    value = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(longitude_delta / 2) ** 2
    )
    return radius_km * 2 * math.atan2(
        math.sqrt(value), math.sqrt(max(0.0, 1 - value))
    )


def sort_trajectory_by_utc(trajectory: pd.DataFrame) -> pd.DataFrame:
    if "UTC_time" not in trajectory.columns or len(trajectory) < 2:
        return trajectory.copy()
    parsed = (
        trajectory["_utc_datetime"]
        if "_utc_datetime" in trajectory.columns
        else pd.to_datetime(
            trajectory["UTC_time"], format="mixed", utc=True, errors="raise"
        )
    )
    ordered = trajectory.assign(_sort_time=parsed).sort_values(
        "_sort_time", kind="stable", na_position="last"
    )
    return ordered.drop(columns="_sort_time")


def prepare_trajectory_dataframe(
    dataframe: pd.DataFrame,
    require_geo: bool = False,
) -> pd.DataFrame:
    required = {
        "trajectory_id", "user_id", "POI_id", "POI_catname", "UTC_time",
        "time_period",
    }
    if require_geo:
        required.update({"latitude", "longitude", "timezone"})
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"trajectory data missing columns: {missing}")
    null_columns = sorted(
        column for column in required if dataframe[column].isna().any()
    )
    if null_columns:
        raise ValueError(f"trajectory data contains null values in: {null_columns}")
    prepared = dataframe.copy()
    parsed = pd.to_datetime(
        prepared["UTC_time"], format="mixed", utc=True, errors="coerce"
    )
    if parsed.isna().any():
        examples = prepared.loc[parsed.isna(), "UTC_time"].astype(str).head(5).tolist()
        raise ValueError(f"unparseable UTC_time values: {examples}")
    prepared["_utc_datetime"] = parsed
    if require_geo:
        for column in ("latitude", "longitude", "timezone"):
            values = pd.to_numeric(prepared[column], errors="coerce")
            invalid = ~np.isfinite(values.to_numpy(dtype=float))
            if invalid.any():
                raise ValueError(f"column {column} contains {int(invalid.sum())} invalid values")
            prepared[column] = values
    return prepared.sort_values(
        ["trajectory_id", "_utc_datetime"], kind="stable"
    ).reset_index(drop=True)


def _hour_bucket(hour: int, width: int) -> str:
    start = (int(hour) // int(width)) * int(width)
    return f"{start:02d}-{min(start + int(width), 24):02d}"


def _scenario(categories: tuple[str, ...]) -> str:
    normalized = tuple(str(category).lower() for category in categories)
    keyword_groups = (
        (
            "mobility",
            (
                "station", "subway", "rail", "airport", "bus", "taxi",
                "ferry", "travel", "transport",
            ),
        ),
        ("work_study", ("office", "building", "school", "university", "college", "library")),
        ("home", ("home", "residential", "apartment", "housing")),
        ("food_social", ("restaurant", "shop", "café", "cafe", "bakery", "bar", "food", "deli")),
        ("outdoor", ("park", "plaza", "outdoors", "recreation", "beach", "garden", "river")),
    )
    for name, keywords in keyword_groups:
        if any(keyword in category for category in normalized for keyword in keywords):
            return name
    return "general"


def _key_from_parts(categories: tuple[str, ...], hour: int, weekend: bool) -> str:
    preview = tuple(
        str(category) for category in categories[-SCENARIO_PREVIEW_LENGTH:]
    )
    scenario = _scenario(preview)
    suffix_length = 2 if scenario in {"mobility", "home"} else 3
    bucket_width = 1 if scenario == "work_study" else 2
    suffix = preview[-min(suffix_length, len(preview)):]
    day_type = "weekend" if weekend else "weekday"
    return f"{day_type}|{_hour_bucket(hour, bucket_width)}|{scenario}|{'>'.join(suffix)}"


def canonical_key(prefix: pd.DataFrame) -> str:
    if len(prefix) == 0:
        raise ValueError("cannot build a support key from an empty prefix")
    categories = tuple(
        str(value)
        for value in prefix["POI_catname"].iloc[-SCENARIO_PREVIEW_LENGTH:]
    )
    last_row = prefix.iloc[-1]
    parsed = _parse_utc_time(last_row["UTC_time"], _row_timezone_minutes(last_row))
    return _key_from_parts(categories, int(parsed["hour"]), bool(parsed["weekend"]))


def canonical_keys_for_prediction_steps(trajectory: pd.DataFrame) -> tuple[str, ...]:
    if len(trajectory) == 0:
        return ()
    categories = tuple(str(value) for value in trajectory["POI_catname"].tolist())
    utc_datetimes = (
        tuple(trajectory["_utc_datetime"].tolist())
        if "_utc_datetime" in trajectory.columns
        else tuple(pd.to_datetime(
            trajectory["UTC_time"], format="mixed", utc=True, errors="raise"
        ).tolist())
    )
    offsets = (
        tuple(trajectory["timezone"].tolist())
        if "timezone" in trajectory.columns
        else (0,) * len(trajectory)
    )
    keys = []
    for position, (utc_datetime, offset) in enumerate(zip(utc_datetimes, offsets)):
        local_datetime = utc_datetime + pd.Timedelta(
            minutes=_coerce_timezone_minutes(offset)
        )
        preview = categories[
            max(0, position + 1 - SCENARIO_PREVIEW_LENGTH):position + 1
        ]
        keys.append(_key_from_parts(
            preview,
            hour=int(local_datetime.hour),
            weekend=int(local_datetime.dayofweek) >= 5,
        ))
    return tuple(keys)


def build_system_prompt() -> str:
    categories = list(ACTIVITY12_CATEGORIES)
    definitions = {
        category: ACTIVITY12_DESCRIPTIONS[category] for category in categories
    }
    return (
        "Estimate the next activity-class probability distribution from an "
        "observed trajectory prefix.\n"
        "Use only general activity-transition knowledge to interpret the observed prefix. "
        "Do not invent user attributes, unobserved locations, or the unknown next-visit time, "
        "venue, or category. Treat the labels as mutually exclusive and classify each venue "
        "by its primary function using LABEL DEFINITIONS.\n"
        f"Return only one JSON object with exactly {len(categories)} distinct keys copied "
        "verbatim from TAXONOMY. Values must be finite non-negative relative-likelihood "
        "weights, with at least one positive value; they need not sum to one.\n"
        f"TAXONOMY={json.dumps(categories, ensure_ascii=False, separators=(',', ':'))}\n"
        f"LABEL DEFINITIONS={json.dumps(definitions, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_trajectory_prompt(prefix: pd.DataFrame) -> str:
    lines = []
    preceding_row = None
    preceding_time = None
    for index, (_, row) in enumerate(prefix.iterrows(), start=1):
        parsed = _parse_utc_time(row["UTC_time"], _row_timezone_minutes(row))
        fine_category = str(row["POI_catname"])
        activity = fine_to_activity12(fine_category)
        fields = [str(index), parsed["time_str"], f"{fine_category} [{activity}]"]
        if preceding_row is not None:
            elapsed = _format_elapsed(preceding_time, parsed["utc_datetime"])
            distance = _haversine_km(preceding_row, row)
            if elapsed is not None:
                fields.append(f"after={elapsed}")
            if distance is not None:
                fields.append(f"move={distance:.2f}km")
        lines.append(" | ".join(fields))
        preceding_row = row
        preceding_time = parsed["utc_datetime"]
    return (
        "OBSERVED PREFIX (times are local; next-visit time is unknown):\n"
        + "\n".join(lines)
        + "\nPredict the next activity-class distribution."
    )


def validate_probability_distribution(raw: dict) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("teacher output must be a JSON object")
    validated = {}
    for category, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"non-numeric teacher weight for {category!r}")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid teacher weight for {category!r}: {value!r}")
        validated[str(category)] = value
    mass = sum(validated.values())
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("teacher weight mass must be finite and positive")
    return validated


def normalize_activity12_distribution(raw: dict) -> dict[str, float]:
    validated = validate_probability_distribution(raw)
    expected = list(ACTIVITY12_CATEGORIES)
    if set(validated) != set(expected) or len(validated) != len(expected):
        missing = sorted(set(expected) - set(validated))
        extra = sorted(set(validated) - set(expected))
        raise ValueError(f"teacher keys must equal Activity12; missing={missing}, extra={extra}")
    mass = sum(validated.values())
    return {category: validated[category] / mass for category in expected}


class TeacherCache:
    def __init__(
        self,
        cache_path: str = "data/NYC/activity12_teacher.json",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com/v1",
        max_retries: int = 2,
        llm_temperature: float = 0.05,
        llm_max_tokens: int = 1024,
        timeout: float = 30.0,
        pool_timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        use_env_proxy: bool = True,
        taxonomy_fingerprint: str | None = None,
    ):
        fingerprint = activity12_mapping_sha256()
        if taxonomy_fingerprint not in {None, fingerprint}:
            raise ValueError("Activity12 mapping fingerprint mismatch")
        self.cache_path = str(cache_path)
        self.llm_temperature = float(llm_temperature)
        self.llm_max_tokens = int(llm_max_tokens)
        self.taxonomy_fingerprint = fingerprint
        self.entries: dict[str, dict] = {}
        self.invalid_reasons = Counter()
        self.client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            pool_timeout=pool_timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            use_env_proxy=use_env_proxy,
        )

    def generation_config(self) -> dict:
        prompt_source = "".join(inspect.getsource(function) for function in (
            build_system_prompt,
            build_trajectory_prompt,
            normalize_activity12_distribution,
        ))
        return {
            "schema": CACHE_SCHEMA,
            "prompt_schema": "observed_prefix_activity12",
            "prompt_sha256": hashlib.sha256(prompt_source.encode("utf-8")).hexdigest(),
            "teacher_model": TEACHER_MODEL,
            "teacher_base_url": self.client.base_url,
            "taxonomy_name": "activity12",
            "taxonomy_mapping_sha256": self.taxonomy_fingerprint,
            "category_count": len(ACTIVITY12_CATEGORIES),
            "scenario_preview_length": SCENARIO_PREVIEW_LENGTH,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "include_supported": False,
        }

    def generation_fingerprint(self) -> str:
        payload = json.dumps(
            self.generation_config(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def training_key_counts(dataframe: pd.DataFrame) -> Counter:
        counts = Counter()
        for _, trajectory in dataframe.groupby("trajectory_id", sort=True):
            trajectory = sort_trajectory_by_utc(trajectory)
            if len(trajectory) < 2:
                continue
            counts.update(canonical_keys_for_prediction_steps(trajectory)[:-1])
        return counts

    def _query_prefix(self, prefix: pd.DataFrame) -> dict[str, float]:
        raw = self.client.query(
            build_trajectory_prompt(prefix),
            system_prompt=build_system_prompt(),
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )
        return normalize_activity12_distribution(raw)

    def _entry(
        self,
        trajectory_id: str,
        trajectory: pd.DataFrame,
        key: str,
        support_without_self: int,
        distribution: dict[str, float],
        error: str = "",
    ) -> dict:
        target = trajectory.iloc[-1]
        return {
            "trajectory_id": str(trajectory_id),
            "distribution": distribution,
            "true_category": fine_to_activity12(str(target["POI_catname"])),
            "pattern_key": key,
            "support_without_self": int(support_without_self),
            "valid": bool(distribution),
            "cache_schema": CACHE_SCHEMA,
            "generation_fingerprint": self.generation_fingerprint(),
            "generation_config": self.generation_config(),
            "error": str(error),
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        temporary = self.cache_path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(self.entries, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.cache_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def load(self) -> dict[str, dict]:
        with open(self.cache_path, "r", encoding="utf-8") as handle:
            cache = json.load(handle)
        required = {
            "trajectory_id", "distribution", "true_category", "pattern_key",
            "support_without_self", "valid", "cache_schema",
            "generation_fingerprint", "generation_config", "error",
        }
        for trajectory_id, entry in cache.items():
            missing = required.difference(entry)
            if missing:
                raise ValueError(
                    f"cache entry {trajectory_id!r} is missing fields: "
                    f"{sorted(missing)}"
                )
            if entry["cache_schema"] != CACHE_SCHEMA:
                raise ValueError(f"cache entry {trajectory_id!r} has an invalid schema")
            if entry["valid"]:
                entry["distribution"] = normalize_activity12_distribution(
                    entry["distribution"]
                )
            elif entry["distribution"]:
                raise ValueError(f"invalid cache entry {trajectory_id!r} contains a distribution")
        self.entries = cache
        return cache

    def generate(
        self,
        csv_path: str,
        resume: bool = True,
        workers: int = 20,
    ) -> dict[str, dict]:
        dataframe = prepare_trajectory_dataframe(
            pd.read_csv(csv_path), require_geo=True
        )
        validate_activity12_coverage(dataframe["POI_catname"].astype(str).unique())
        counts = self.training_key_counts(dataframe)
        fingerprint = self.generation_fingerprint()
        if resume and os.path.exists(self.cache_path):
            self.load()
            if any(
                entry["generation_fingerprint"] != fingerprint
                for entry in self.entries.values()
            ):
                raise ValueError("cache fingerprint differs from the active configuration")

        pending = []
        for trajectory_id, trajectory in dataframe.groupby("trajectory_id", sort=True):
            trajectory = sort_trajectory_by_utc(trajectory)
            prefix = trajectory.iloc[:-1]
            if len(prefix) < 1:
                continue
            key = canonical_key(prefix)
            support_without_self = max(0, int(counts[key]) - 1)
            if support_without_self != 0:
                continue
            existing = self.entries.get(str(trajectory_id))
            if existing and existing["valid"]:
                continue
            pending.append((
                str(trajectory_id), trajectory, key, support_without_self
            ))

        def query(record):
            trajectory_id, trajectory, key, support_without_self = record
            try:
                distribution = self._query_prefix(trajectory.iloc[:-1])
                return trajectory_id, self._entry(
                    trajectory_id, trajectory, key, support_without_self,
                    distribution,
                )
            except Exception as exc:
                error = str(exc).strip()[:300] or "teacher_query_failed"
                return trajectory_id, self._entry(
                    trajectory_id, trajectory, key, support_without_self, {}, error
                )

        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [executor.submit(query, record) for record in pending]
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Generating Activity12 distributions",
            ):
                trajectory_id, entry = future.result()
                self.entries[trajectory_id] = entry
                if not entry["valid"]:
                    self.invalid_reasons[entry["error"]] += 1
                completed += 1
                if completed % 20 == 0:
                    self.save()

        self.save()
        valid = sum(entry["valid"] for entry in self.entries.values())
        print(
            f"[TeacherCache] path={self.cache_path}, requested={len(pending)}, "
            f"valid={valid}, entries={len(self.entries)}"
        )
        if self.invalid_reasons:
            print(f"[TeacherCache] invalid={self.invalid_reasons.most_common(10)}")
        return self.entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train", default="data/NYC/NYC_train.csv")
    parser.add_argument("--cache_path", default="data/NYC/activity12_teacher.json")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--pool_timeout", type=float, default=30.0)
    parser.add_argument("--max_connections", type=int, default=100)
    parser.add_argument("--max_keepalive_connections", type=int, default=20)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--llm_temperature", type=float, default=0.05)
    parser.add_argument("--llm_max_tokens", type=int, default=1024)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--disable_env_proxy", action="store_true")
    args = parser.parse_args()

    cache = TeacherCache(
        cache_path=args.cache_path,
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        timeout=args.timeout,
        pool_timeout=args.pool_timeout,
        max_connections=args.max_connections,
        max_keepalive_connections=args.max_keepalive_connections,
        use_env_proxy=not args.disable_env_proxy,
        taxonomy_fingerprint=activity12_mapping_sha256(),
    )
    cache.generate(
        args.data_train,
        resume=not args.no_resume,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
