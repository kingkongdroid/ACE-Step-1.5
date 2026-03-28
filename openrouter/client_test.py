"""
ACE-Step OpenRouter API client test code

Tests various endpoints and feature modes of the API using the requests library.

Usage:
    python -m openrouter.test_client
    python -m openrouter.test_client --base-url http://127.0.0.1:8002
    python -m openrouter.test_client --api-key your-api-key
"""

import argparse
import base64
import json
import os
import sys
import time
from typing import Optional

import requests

# This file is an executable API client script, not a pytest test module.
__test__ = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_URL = "https://api.acemusic.ai"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")


def get_headers(api_key: Optional[str] = None) -> dict:
    """Build request headers"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def handle_response(resp, audio_filename: str) -> bool:
    """Handle API response and save audio"""
    print(f"Status code: {resp.status_code}")

    if resp.status_code != 200:
        print(f"Error response: {resp.text}")
        return False

    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    audio_list = message.get("audio") or []

    print(f"\nContent:\n{content if content else '(no text content)'}")
    print(f"Audio count: {len(audio_list)}")

    if audio_list and len(audio_list) > 0:
        audio_url = audio_list[0].get("audio_url", {}).get("url", "")
        if audio_url:
            filepath = save_audio(audio_url, audio_filename)
            print(f"Audio saved: {filepath}")
        else:
            print("Warning: audio_url is empty")
    else:
        print("Warning: no audio data returned")

    return True


def save_audio(audio_url: str, filename: str) -> str:
    """Save audio file from base64 data URL"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse data URL: data:audio/mpeg;base64,<data>
    if not audio_url.startswith("data:"):
        print(f"  [Warning] Invalid audio URL format")
        return ""

    # Extract base64 data
    b64_data = audio_url.split(",", 1)[1]
    audio_bytes = base64.b64decode(b64_data)

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)

    return filepath


# =============================================================================
# Test functions
# =============================================================================

def test_health(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test the health check endpoint"""
    print("\n" + "=" * 60)
    print("Test: GET /health")
    print("=" * 60)

    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status code: {resp.status_code}")
        print(f"Response: {json.dumps(resp.text, indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_list_models(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test the model list endpoint"""
    print("\n" + "=" * 60)
    print("Test: GET /api/v1/models")
    print("=" * 60)

    try:
        resp = requests.get(
            f"{base_url}/api/v1/models",
            headers=get_headers(api_key),
            timeout=10
        )
        print(f"Status code: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_natural_language_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test natural language mode (Sample Mode)"""
    print("\n" + "=" * 60)
    print("Test: Natural Language Mode (Sample Mode)")
    print("=" * 60)

    payload = {
        "messages": [
            {"role": "user", "content": "Generate an upbeat pop song about summer and travel"}
        ],
        "sample_mode": True,
        "audio_config": {
            "vocal_language": "en",
            "duration": 30,
        },
    }

    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"Status code: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            audio_list = message.get("audio") or []

            print(f"\nContent:\n{content if content else '(no text content)'}")
            print(f"Audio count: {len(audio_list)}")

            if audio_list and len(audio_list) > 0:
                audio_item = audio_list[0]
                audio_url = audio_item.get("audio_url", {}).get("url", "")
                if audio_url:
                    filepath = save_audio(audio_url, "test_natural_language.mp3")
                    print(f"Audio saved: {filepath}")
                else:
                    print("Warning: audio_url is empty")
            else:
                print("Warning: no audio data returned")

            return True
        else:
            print(f"Error response: {resp.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tagged_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test tagged mode (Tagged Mode)"""
    print("\n" + "=" * 60)
    print("Test: Tagged Mode (Tagged Mode)")
    print("=" * 60)

    content = """<prompt>A gentle acoustic ballad in C major, 80 BPM, female vocal</prompt>
<lyrics>[Verse 1]
Sunlight through the window
A brand new day begins

[Chorus]
We are the dreamers
We are the light</lyrics>"""

    payload = {
        "messages": [{"role": "user", "content": content}],
        "audio_config": {
            "vocal_language": "en",
            "duration": 30,
        },
    }

    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"Status code: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            audio_list = message.get("audio") or []

            print(f"\nContent:\n{content if content else '(no text content)'}")
            print(f"Audio count: {len(audio_list)}")

            if audio_list and len(audio_list) > 0:
                audio_url = audio_list[0].get("audio_url", {}).get("url", "")
                if audio_url:
                    filepath = save_audio(audio_url, "test_tagged_mode.mp3")
                    print(f"Audio saved: {filepath}")

            return True
        else:
            print(f"Error response: {resp.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lyrics_only_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test lyrics-only mode (Lyrics-Only Mode)"""
    print("\n" + "=" * 60)
    print("Test: Lyrics-Only Mode (Lyrics-Only Mode)")
    print("=" * 60)

    lyrics = """[Verse 1]
Walking down the street
Feeling the beat

[Chorus]
Dance with me tonight
Under the moonlight"""

    payload = {
        "messages": [{"role": "user", "content": lyrics}],
        "audio_config": {
            "vocal_language": "en",
            "duration": 30,
        },
    }

    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"Status code: {resp.status_code}")

        return handle_response(resp, "test_lyrics_only.mp3")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instrumental_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test instrumental mode (Instrumental Mode)"""
    print("\n" + "=" * 60)
    print("Test: Instrumental Mode (Instrumental Mode)")
    print("=" * 60)

    payload = {
        "messages": [
            {"role": "user", "content": "<prompt>Epic orchestral cinematic score, dramatic and powerful</prompt>"}
        ],
        "audio_config": {
            "instrumental": True,
            "duration": 30,
        },
    }

    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"Status code: {resp.status_code}")

        return handle_response(resp, "test_instrumental.mp3")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_mode(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test streaming response mode (Streaming Mode)"""
    print("\n" + "=" * 60)
    print("Test: Streaming Response Mode (Streaming Mode)")
    print("=" * 60)

    payload = {
        "messages": [
            {"role": "user", "content": "Generate a cheerful guitar piece"}
        ],
        "stream": True,
        "sample_mode": True,
        "audio_config": {
            "instrumental": True,
            "duration": 30,
        },
    }

    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            stream=True,
            timeout=300
        )
        print(f"Status code: {resp.status_code}")

        if resp.status_code == 200:
            content_parts = []
            audio_url = None

            print("\nReceiving streaming data:")
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                if line == "data: [DONE]":
                    print("  [DONE]")
                    break

                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    finish_reason = chunk["choices"][0].get("finish_reason")

                    if "role" in delta:
                        print(f"  Role: {delta['role']}")

                    if "content" in delta and delta["content"]:
                        content_parts.append(delta["content"])
                        # Do not print heartbeat dots
                        if delta["content"] != ".":
                            print(f"  Content: {delta['content'][:100]}...")
                        else:
                            print("  [Heartbeat]")

                    if "audio" in delta and delta["audio"]:
                        audio_item = delta["audio"][0]
                        audio_url = audio_item.get("audio_url", {}).get("url", "")
                        if audio_url:
                            print(f"  Audio data received (length: {len(audio_url)} chars)")

                    if finish_reason:
                        print(f"  Finish reason: {finish_reason}")

                except json.JSONDecodeError as e:
                    print(f"  [Parse error] {e}")

            full_content = "".join(content_parts)
            print(f"\nFull content:\n{full_content}")

            if audio_url:
                filepath = save_audio(audio_url, "test_streaming.mp3")
                print(f"\nAudio saved: {filepath}")

            return True
        else:
            print(f"Error response: {resp.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_full_parameters(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test full parameter control"""
    print("\n" + "=" * 60)
    print("Test: Full Parameter Control")
    print("=" * 60)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": "<prompt>Dreamy lo-fi hip hop beat with vinyl crackle</prompt><lyrics>[inst]</lyrics>"
            }
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        "thinking": False,
        "use_cot_caption": True,
        "use_cot_language": False,
        "use_format": True,
        "audio_config": {
            "bpm": 85,
            "duration": 30,
            "instrumental": True,
        },
    }

    print(f"Request: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=300
        )
        print(f"Status code: {resp.status_code}")

        return handle_response(resp, "test_full_params.mp3")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling(base_url: str, api_key: Optional[str] = None) -> bool:
    """Test error handling"""
    print("\n" + "=" * 60)
    print("Test: Error Handling")
    print("=" * 60)

    # Test empty messages
    print("\n1. Test empty messages:")
    payload = {"messages": []}
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=30
        )
        print(f"  Status code: {resp.status_code}")
        print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test empty content message
    print("\n2. Test empty content message:")
    payload = {"messages": [{"role": "user", "content": ""}]}
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=get_headers(api_key),
            json=payload,
            timeout=30
        )
        print(f"  Status code: {resp.status_code}")
        print(f"  Response: {resp.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

    return True


# =============================================================================
# Main function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ACE-Step OpenRouter API client test")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="API key (optional)"
    )
    parser.add_argument(
        "--test",
        choices=[
            "health", "models", "natural", "tagged", "lyrics",
            "instrumental", "streaming", "full", "error", "all"
        ],
        default="health",
        help="Test to run (default: health)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ACE-Step OpenRouter API Client Test")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"API Key: {'set' if args.api_key else 'not set'}")
    print(f"Output directory: {OUTPUT_DIR}")

    tests = {
        "health": test_health,
        "models": test_list_models,
        "natural": test_natural_language_mode,
        "tagged": test_tagged_mode,
        "lyrics": test_lyrics_only_mode,
        "instrumental": test_instrumental_mode,
        "streaming": test_streaming_mode,
        "full": test_full_parameters,
        "error": test_error_handling,
    }

    results = {}

    if args.test == "all":
        for name, test_func in tests.items():
            results[name] = test_func(args.base_url, args.api_key)
    else:
        results[args.test] = tests[args.test](args.base_url, args.api_key)

    # Print test results summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    # Return exit code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
