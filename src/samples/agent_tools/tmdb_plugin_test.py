# test_tmdb_plugin.py
import asyncio
import json
import os

from dotenv import load_dotenv
from semantic_kernel import Kernel

from src.samples.agent_tools.plugins.tmdb import TMDbService

load_dotenv()


def _ensure_token():
    tok = os.environ.get("TMDB_BEARER_TOKEN")
    if not tok:
        raise RuntimeError(
            "TMDB_BEARER_TOKEN env var not set.\n"
            "Get a TMDb v4 Read Access Token and export it, e.g.:\n"
            '  export TMDB_BEARER_TOKEN="eyJhbGciOi..."'
        )


async def _register_tmdb_plugin(
    kernel: "Kernel", tmdb: TMDbService, plugin_name: str = "tmdb"
):
    """
    Works with both newer and older SK Python APIs.
    Returns a dict-like plugin container exposing 'get_movie_genre_id' etc.
    """
    # Newer SK often exposes: kernel.add_plugin(object, plugin_name)
    if hasattr(kernel, "add_plugin"):
        return kernel.add_plugin(tmdb, plugin_name=plugin_name)

    raise RuntimeError(
        "Could not find a compatible way to add a native plugin to Semantic Kernel. "
        "Please upgrade semantic-kernel."
    )


async def _invoke(kernel: "Kernel", plugin, func_name: str, **kwargs):
    """
    Invoke a kernel function across SK versions.
    """
    # Most SK versions give a dict-like plugin: plugin["function_name"]
    fn = plugin[func_name]

    if fn is None:
        raise RuntimeError(f"Could not resolve function '{func_name}' from the plugin.")

    # Newer SK: await kernel.invoke(fn, **kwargs)
    if hasattr(kernel, "invoke"):
        result = await kernel.invoke(fn, **kwargs)
        # result may be a KernelResult or just the payload; normalize to string
        return str(result)

    # Older SK sometimes calls functions directly:
    maybe_coro = fn(**kwargs)
    if asyncio.iscoroutine(maybe_coro):
        return await maybe_coro
    return maybe_coro


async def main():
    _ensure_token()

    # 1) Build SK kernel and plugin
    kernel = Kernel()
    tmdb = TMDbService()  # reads env vars by default
    plugin = kernel.add_plugin(tmdb, plugin_name="TMDbService")

    # 2) Call: get_movie_genre_id
    genre_name = "Action"
    raw_genre_id = await _invoke(
        kernel, plugin, "get_movie_genre_id", genre_name=genre_name
    )

    genre_id = int(str(raw_genre_id).strip())

    assert isinstance(genre_id, int), "Genre id should be an integer"
    print(f"[OK] get_movie_genre_id('{genre_name}') -> {genre_id}")

    # 3) Call: get_top_movies_by_genre
    raw_movies = await _invoke(
        kernel, plugin, "get_top_movies_by_genre", genre_name=genre_name
    )

    # Your plugin returns a JSON string; parse and validate
    movies = json.loads(str(raw_movies))
    assert isinstance(movies, list), "Expected a list of movies"
    print(f"[OK] get_top_movies_by_genre('{genre_name}') -> {len(movies)} results")

    # Show a few nicely
    for m in movies[:5]:
        title = m.get("title")
        year = (m.get("release_date") or "")[:4]
        rating = m.get("vote_average")
        votes = m.get("vote_count")
        print(f"  - {title} ({year}) ★{rating} [{votes} votes]")


if __name__ == "__main__":
    asyncio.run(main())
