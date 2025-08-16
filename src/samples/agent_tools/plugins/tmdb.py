# --- Standard library / deps are imported inside the file to keep the class top line as requested ---
import os
import time
from dataclasses import dataclass

import requests
from semantic_kernel.functions import kernel_function


class TMDbService:
    """
    Semantic Kernel plugin for querying TMDb (The Movie Database).

    Setup:
      1) Create a TMDb API Read Access Token (v4 auth) at https://www.themoviedb.org/settings/api
      2) Export it as an environment variable:
           export TMDB_BEARER_TOKEN="eyJhbGciOiJIUzI1NiIsInR..."
      3) (Optional) Set a default language/region:
           export TMDB_LANGUAGE="en-US"
           export TMDB_REGION="US"

    Usage with Semantic Kernel (Python):
      from semantic_kernel import Kernel
      from tmdb_plugin import TMDbService

      kernel = Kernel()
      tmdb = TMDbService()
      kernel.add_plugin(tmdb, plugin_name="tmdb")

      # Then you can call exposed functions by name in planners or prompts:
      # - tmdb.get_movie_genre_id
      # - tmdb.get_top_movies_by_genre
    """

    _BASE_URL = "https://api.themoviedb.org/3"
    _GENRE_ENDPOINT = "/genre/movie/list"
    _DISCOVER_ENDPOINT = "/discover/movie"

    @dataclass
    class _Config:
        bearer_token: str
        language: str = os.environ.get("TMDB_LANGUAGE", "en-US")
        region: str | None = os.environ.get("TMDB_REGION") or None
        timeout_sec: int = int(os.environ.get("TMDB_TIMEOUT_SEC", "15"))
        # Simple retry settings
        max_retries: int = int(os.environ.get("TMDB_MAX_RETRIES", "3"))
        backoff_sec: float = float(os.environ.get("TMDB_RETRY_BACKOFF_SEC", "0.75"))

    def __init__(
        self,
        bearer_token: str | None = None,
        language: str | None = None,
        region: str | None = None,
    ):
        """
        Optionally pass bearer_token/language/region directly. Otherwise reads from env:
          TMDB_BEARER_TOKEN (required), TMDB_LANGUAGE, TMDB_REGION
        """
        token = bearer_token or os.environ.get("TMDB_BEARER_TOKEN")
        if not token:
            raise ValueError(
                "TMDB_BEARER_TOKEN is not set. Please export a TMDb v4 API Read Access Token."
            )
        self.config = self._Config(
            bearer_token=token,
            language=language or os.environ.get("TMDB_LANGUAGE", "en-US"),
            region=region or (os.environ.get("TMDB_REGION") or None),
        )

    # ------------------------- Internal helpers -------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.bearer_token}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self._BASE_URL}{path}"
        params = dict(params or {})
        # Language/region defaulting
        if "language" not in params and self.config.language:
            params["language"] = self.config.language
        if self.config.region and "region" not in params:
            params["region"] = self.config.region

        last_err = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=self.config.timeout_sec,
                )
                if resp.status_code == 429:
                    # Rate limited — respect Retry-After if present
                    retry_after = float(
                        resp.headers.get(
                            "Retry-After", self.config.backoff_sec * attempt
                        )
                    )
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                if attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.backoff_sec * attempt)
        raise RuntimeError(f"TMDb request failed after retries: {last_err}")

    def _normalize_genre(self, name: str) -> str:
        return name.strip().lower()

    # ------------------------- Exposed SK functions -------------------------

    @kernel_function(
        name="get_movie_genre_id",
        description="Return the TMDb numeric genre id for a given movie genre name. Case-insensitive.",
    )
    def get_movie_genre_id(self, genre_name: str) -> int:
        """
        Args:
            genre_name: A movie genre name such as 'Action', 'Comedy', 'Science Fiction', 'Animation', etc.

        Returns:
            The integer TMDb genre id.

        Raises:
            ValueError if the genre cannot be found.
        """
        if not genre_name or not genre_name.strip():
            raise ValueError("genre_name cannot be empty")

        data = self._get(self._GENRE_ENDPOINT)
        genres = data.get("genres", [])

        # Build map with case-insensitive keys and common aliases
        name_to_id: dict[str, int] = {}
        for g in genres:
            n = (g.get("name") or "").strip()
            gid = int(g.get("id"))
            if n:
                name_to_id[self._normalize_genre(n)] = gid

        # Add a few helpful aliases frequently used
        aliases = {
            "sci-fi": "science fiction",
            "scifi": "science fiction",
            "romcom": "romance",
            "kids": "family",
            "doc": "documentary",
            "biopic": "history",
        }
        for alias, canonical in aliases.items():
            if canonical in name_to_id:
                name_to_id[alias] = name_to_id[canonical]

        key = self._normalize_genre(genre_name)
        if key not in name_to_id:
            # Try fuzzy-ish contains match
            for k in name_to_id:
                if key in k or k in key:
                    return name_to_id[k]
            raise ValueError(
                f"Unknown genre: '{genre_name}'. Available: {', '.join(sorted(name_to_id))}"
            )

        return name_to_id[key]

    @kernel_function(
        name="get_top_movies_by_genre",
        description=(
            "Return a concise list of top-rated movies for a given genre name. "
            "Uses TMDb Discover API with sensible filters."
        ),
    )
    def get_top_movies_by_genre(self, genre_name: str) -> str:
        """
        Args:
            genre_name: Human-readable genre (e.g., 'Action', 'Comedy', 'Science Fiction').

        Returns:
            A JSON string representing a list of up to 10 top movies for the given genre.
            Each item includes: id, title, release_date, vote_average, vote_count, overview, original_language.

        Notes:
            - Filters out very low-vote titles with `vote_count.gte=200` (tweak as needed).
            - Sorts by `vote_average.desc` primarily; if you prefer popularity, change sort_by.
        """
        import json

        genre_id = self.get_movie_genre_id(genre_name)

        params = {
            "with_genres": str(genre_id),
            "sort_by": "vote_average.desc",
            "vote_count.gte": 200,
            "include_adult": "false",
            "page": 1,
        }

        data = self._get(self._DISCOVER_ENDPOINT, params=params)
        results = data.get("results", [])[:10]

        simplified = []
        for r in results:
            simplified.append(
                {
                    "id": r.get("id"),
                    "title": r.get("title") or r.get("name"),
                    "release_date": r.get("release_date"),
                    "vote_average": r.get("vote_average"),
                    "vote_count": r.get("vote_count"),
                    "original_language": r.get("original_language"),
                    "overview": r.get("overview"),
                }
            )

        return json.dumps(simplified, ensure_ascii=False, indent=2)
