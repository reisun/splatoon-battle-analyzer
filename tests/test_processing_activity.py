"""Actual execution admission and reconciliation contract."""

import asyncio
import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import api
from src.activity import ProcessingActivity


def test_instance_identity_changes():
    assert ProcessingActivity().instance_id != ProcessingActivity().instance_id


def test_atomic_claim_and_stale_release():
    activity = ProcessingActivity()
    first = activity.claim("scan")
    assert activity.claim("highlights") is None
    activity.release(first)
    second = activity.claim("highlights")
    activity.release(first)
    assert activity.snapshot() is not None
    activity.release(second)


def test_simultaneous_claims_allow_only_one():
    activity = ProcessingActivity()
    barrier = threading.Barrier(8)
    results = []

    def claim():
        barrier.wait(timeout=2)
        results.append(activity.claim("scan"))

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert len(results) == 8
    assert sum(result is not None for result in results) == 1


def test_active_worker_rejects_all_routes_without_creating_job(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")

    def worker(job_id, request):
        api.job_store.mark_running(job_id)
        api.job_store.update_progress(job_id, 1, 3, 10)
        entered.set()
        assert finish.wait(5)
        api.job_store.mark_completed(job_id, {})

    client = TestClient(api.app)
    with (
        patch("src.api.check_api_key_available", return_value=True),
        patch("src.api._run_scan_job", side_effect=worker),
    ):
        response = client.post("/analyze/matches/scan/jobs", json={"file_path": str(video)})
        assert entered.wait(2)
        try:
            current = client.get("/processing").json()
            assert current["busy"] is True
            assert current["operations"][0]["job_id"] == response.json()["job_id"]
            assert current["operations"][0]["progress"]["frames_done"] == 3
            with patch.object(api.job_store, "create") as create:
                for endpoint in (
                    "/analyze/highlights",
                    "/analyze/highlights/jobs",
                    "/analyze/matches/scan/jobs",
                ):
                    assert client.post(endpoint, json={"file_path": str(video)}).status_code == 409
                create.assert_not_called()
        finally:
            finish.set()
            api._executor.submit(lambda: None).result(timeout=2)
    assert client.get("/processing").json()["busy"] is False


def test_worker_exception_releases_claim():
    def fail(job_id, request):
        raise RuntimeError("failed")

    future = api._submit_operation("scan", fail, None)
    with pytest.raises(RuntimeError, match="failed"):
        future.result(timeout=2)
    assert api.activity.snapshot() is None
    assert api.job_store.get(future.job_id).status.value == "failed"


def test_cancelled_http_wait_does_not_release_running_worker():
    entered, finish = threading.Event(), threading.Event()

    def worker(job_id, request):
        entered.set()
        assert finish.wait(5)

    async def scenario():
        future = api._submit_operation("highlights", worker, None)

        async def wait():
            await asyncio.shield(asyncio.wrap_future(future))

        task = asyncio.create_task(wait())
        assert entered.wait(2)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert api.activity.snapshot() is not None
        finish.set()
        await asyncio.wrap_future(future)
        assert api.activity.snapshot() is None

    try:
        asyncio.run(scenario())
    finally:
        finish.set()
