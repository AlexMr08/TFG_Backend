import os
import random

from locust import HttpUser, between, task


IMAGES_PATH = os.getenv("IMAGES_PATH", "/view")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "").strip()
ITEMS_PER_PAGE = int(os.getenv("ITEMS_PER_PAGE", "20"))
PAGE = int(os.getenv("PAGE", "1"))
ARTIST_ID = os.getenv("ARTIST_ID", "").strip()
STYLE_ID = os.getenv("STYLE_ID", "").strip()
GENRE_ID = os.getenv("GENRE_ID", "").strip()


def _build_params():
    params = {
        "page": PAGE,
        "items_per_page": ITEMS_PER_PAGE,
    }
    if ARTIST_ID:
        params["artist_id"] = ARTIST_ID
    if STYLE_ID:
        params["style_id"] = STYLE_ID
    if GENRE_ID:
        params["genre_id"] = GENRE_ID
    return params


class ImagesListStressUser(HttpUser):
    # Simula un usuario humano que lee la pantalla entre 2 y 5 segundos antes de hacer click
    wait_time = between(2, 5)

    @task
    def fetch_images_once(self):
        if not BEARER_TOKEN:
            raise ValueError("Set BEARER_TOKEN for images stress test")

        headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
        params = _build_params()
        # Sobrescribimos la página para simular que navegan por la galería aleatoriamente
        params["page"] = random.randint(1, 10)

        with self.client.get(
            IMAGES_PATH,
            headers=headers,
            params=params,
            name="GET /view",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:300]}")
            else:
                response.success()
