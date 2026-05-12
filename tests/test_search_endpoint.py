import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.routers.images import imagesRouter, get_current_user_id
from database import get_session, get_chroma_collection


class FakeResult:
    def __init__(self, rows=None, scalar_value=None):
        self._rows = rows or []
        self._scalar_value = scalar_value

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)

    def scalar(self):
        return self._scalar_value


class FakeSession:
    def __init__(self, artists_rows=None, artworks_rows=None, results=None):
        if results is None:
            results = [FakeResult(artists_rows or []), FakeResult(artworks_rows or [])]
        self._results = list(results)
        self.calls = []

    async def execute(self, query, params):
        self.calls.append({"query": getattr(query, "text", str(query)), "params": params})
        return self._results.pop(0)


@pytest.fixture()
def test_app():
    app = FastAPI()
    app.include_router(imagesRouter)
    return app


@pytest.mark.asyncio
async def test_search_empty_query_returns_empty_lists(test_app):
    async def override_get_session():
        yield FakeSession([], [])

    test_app.dependency_overrides[get_session] = override_get_session
    test_app.dependency_overrides[get_chroma_collection] = lambda: None

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/search", params={"query": "   "})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "artists": [],
        "artworks": [],
        "total_artists": 0,
        "total_artworks": 0,
    }


@pytest.mark.asyncio
async def test_search_returns_artists_and_artworks(test_app):
    artist_rows = [
        {"id": "a1", "name": "Claude Monet", "image": "monet.jpg"},
        {"id": "a2", "name": "Camille Pissarro", "image": None},
    ]
    artwork_rows = [
        {
            "id": "i1",
            "name": "Impression, Sunrise",
            "local_route": "monet/sunrise.jpg",
            "year": "1872",
            "artist_id": "a1",
            "style_id": "s1",
            "genre_id": "g1",
            "artist": "Claude Monet",
            "style": "Impressionism",
            "genre": "Landscape",
        }
    ]
    fake_session = FakeSession(artist_rows, artwork_rows)

    async def override_get_session():
        yield fake_session

    test_app.dependency_overrides[get_session] = override_get_session
    test_app.dependency_overrides[get_chroma_collection] = lambda: None

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/search",
            params={"query": "Monet", "artists_limit": 2, "artworks_limit": 1},
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["total_artists"] == 2
    assert payload["total_artworks"] == 1
    assert payload["artists"][0]["name"] == "Claude Monet"
    assert payload["artists"][0]["image_url"] == "monet.jpg"
    assert payload["art"][0]["image_url"] == "/art/monet/sunrise.jpg"

    assert fake_session.calls[0]["params"]["limit"] == 2
    assert fake_session.calls[1]["params"]["limit"] == 1


@pytest.mark.asyncio
async def test_artists_endpoint_includes_top_genre_and_style(test_app):
    artist_rows = [
        {
            "id": "a1",
            "name": "Claude Monet",
            "image": "monet.jpg",
            "genre": "Landscape",
            "style": "Impressionism",
        }
    ]
    fake_session = FakeSession(results=[FakeResult(artist_rows), FakeResult(scalar_value=1)])

    async def override_get_session():
        yield fake_session

    test_app.dependency_overrides[get_session] = override_get_session
    test_app.dependency_overrides[get_current_user_id] = lambda: "test-user"

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/artists", params={"page": 1, "items_per_page": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["artists"][0]["genre"] == "Landscape"
    assert payload["artists"][0]["style"] == "Impressionism"


@pytest.mark.asyncio
async def test_recommended_artists_endpoint_includes_top_genre_and_style(test_app):
    artist_rows = [
        {
            "id": "a1",
            "name": "Claude Monet",
            "image": "monet.jpg",
            "genre": "Landscape",
            "style": "Impressionism",
        }
    ]
    fake_session = FakeSession(results=[FakeResult(artist_rows)])

    async def override_get_session():
        yield fake_session

    test_app.dependency_overrides[get_session] = override_get_session
    test_app.dependency_overrides[get_current_user_id] = lambda: "test-user"

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/recomendedArtists", params={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["artists"][0]["genre"] == "Landscape"
    assert payload["artists"][0]["style"] == "Impressionism"


@pytest.mark.asyncio
async def test_view_returns_paginated_art_with_auth_override(test_app):
    rows = [
        {
            "id": "i1",
            "local_route": "monet/sunrise.jpg",
            "name": "Impression, Sunrise",
            "artist_id": "a1",
            "style_id": "s1",
            "genre_id": "g1",
            "owner_id": None,
            "year": "1872",
            "artist": "Claude Monet",
            "style": "Impressionism",
            "genre": "Landscape",
        },
                {
            "id": "i2",
            "local_route": "monet/sunrise2.jpg",
            "name": "Impression, Sunrise2",
            "artist_id": "a2",
            "style_id": "s2",
            "genre_id": "g2",
            "owner_id": None,
            "year": "1872",
            "artist": "Claude Monet2",
            "style": "Impressionism2",
            "genre": "Landscape2",
        }
    ]
    fake_session = FakeSession(results=[FakeResult(rows), FakeResult(scalar_value=1)])

    async def override_get_session():
        yield fake_session

    test_app.dependency_overrides[get_session] = override_get_session
    test_app.dependency_overrides[get_current_user_id] = lambda: "test-user"

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/view",
            json={"page": 1, "items_per_page": 1, "filtros": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_items"] == 1
    assert payload["art"][0]["image_url"] == "/art/monet/sunrise.jpg"
    assert fake_session.calls[0]["params"]["limit"] == 1
    assert fake_session.calls[0]["params"]["offset"] == 0


@pytest.mark.asyncio
async def test_view_requires_auth(test_app):
    async def override_get_session():
        yield FakeSession(results=[FakeResult([]), FakeResult(scalar_value=0)])

    test_app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/view",
            json={"page": 1, "items_per_page": 1, "filtros": {}},
        )

    assert response.status_code == 401
