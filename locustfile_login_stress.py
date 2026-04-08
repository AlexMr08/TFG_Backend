import os

from locust import HttpUser, constant, task
from locust.exception import StopUser


ARTISTS_PATH = os.getenv("ARTISTS_PATH", "/artists")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "").strip()


class ArtistsListStressUser(HttpUser):
    # No wait: each virtual user performs one request and exits.
    wait_time = constant(0)

    @task
    def fetch_artists_once(self):
        if not BEARER_TOKEN:
            raise ValueError("Set BEARER_TOKEN for artists stress test")

        headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
        with self.client.get(
            ARTISTS_PATH,
            headers=headers,
            name="GET /artists",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:300]}")
            else:
                response.success()

        # Exactly one request per virtual user.
        raise StopUser()
