import json


def extract_json_object(raw_response: str) -> str:
    """Return the first top-level {...} object in raw_response, tolerating surrounding prose."""
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no JSON object found", raw_response, 0)
    return raw_response[start : end + 1]
