"""Swagger UI and the OpenAPI document as served by the running app."""

import json
from typing import Any

import httpx
import pytest

from scheduler_api import openapi

pytestmark = pytest.mark.integration


@pytest.fixture
async def served(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/openapi.json", headers={"X-API-Key": ""})
    assert response.status_code == 200
    document: dict[str, Any] = response.json()
    return document


async def test_docs_page_is_served_without_a_key(client: httpx.AsyncClient) -> None:
    response = await client.get("/docs", headers={"X-API-Key": ""})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "swagger-ui" in response.text
    assert "/openapi.json" in response.text


async def test_served_document_is_the_rendered_one(served: dict[str, Any]) -> None:
    assert served == json.loads(openapi.render())


async def test_declares_the_api_key_security_scheme(served: dict[str, Any]) -> None:
    assert served["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Shared API key configured on the server (`SCHEDULER_API_KEY`).",
    }


async def test_post_requires_idempotency_key_header(served: dict[str, Any]) -> None:
    parameters = served["paths"]["/v1/appointments"]["post"]["parameters"]

    assert {"name": "Idempotency-Key", "in": "header", "required": True}.items() <= next(
        p for p in parameters if p["name"] == "Idempotency-Key"
    ).items()


async def test_post_documents_success_and_error_responses(served: dict[str, Any]) -> None:
    responses = served["paths"]["/v1/appointments"]["post"]["responses"]

    assert {"200", "201", "400", "401", "409", "422"} <= set(responses)
    for status in ("400", "401", "409", "422"):
        assert "application/problem+json" in responses[status]["content"], status


async def test_documented_example_works_against_the_api(
    client: httpx.AsyncClient, served: dict[str, Any]
) -> None:
    """What a user gets by clicking "Try it out" with the defaults."""
    operation = served["paths"]["/v1/appointments"]["post"]
    body = operation["requestBody"]["content"]["application/json"]["examples"]["two_exams"]
    [key] = [p for p in operation["parameters"] if p["name"] == "Idempotency-Key"]

    response = await client.post(
        "/v1/appointments",
        json=body["value"],
        headers={"Idempotency-Key": key["examples"]["example"]["value"]},
    )

    assert response.status_code == 201, response.text
    documented = operation["responses"]["201"]["content"]["application/json"]["example"]
    assert set(response.json()) == set(documented)
