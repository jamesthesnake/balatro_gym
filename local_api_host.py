# balatro_tools.py (on RunPod - modified section)
import requests
import json
import os

# This URL comes from ngrok running on your local PC
LOCAL_GAME_API_BASE_URL = os.environ.get("LOCAL_GAME_API_URL", "http://your-ngrok-url.ngrok-free.app") # Set this env var on RunPod!
LOCAL_GAME_API_KEY = os.environ.get("RUNPOD_TO_LOCAL_API_KEY", "your-secret-key-for-local-game") # Must match key on local server

def _call_local_game_api(endpoint: str, method: str = "GET", data: dict = None):
    url = f"{LOCAL_GAME_API_BASE_URL}{endpoint}"
    headers = {"X-API-Key": LOCAL_GAME_API_KEY, "Content-Type": "application/json"}
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=15)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=15)
        else:
            return {"error": f"Unsupported method: {method}"}
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling local game API at {url}: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from local game API at {url}: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
        return {"error": f"Invalid JSON response from local game API: {e}"}


def check_cards():
    """Return the current 8-card hand by calling the local game API."""
    print("RunPod tool: check_cards() -> calling local game API for hand")
    result = _call_local_game_api("/game/hand")
    return result.get("hand", {"error": "No hand data from local API", "details": result})


def get_optimal_pairs(top_k: int = 5):
    """
    Return best 5-card subsets and their raw-chip scores.
    This example assumes optimal pair calculation can be done on RunPod
    if it has the hand (from check_cards). If it needs live game score
    for the calculation, it would also call a local API endpoint for score.
    """
    print(f"RunPod tool: get_optimal_pairs(top_k={top_k})")
    # First, get the current hand
    current_hand_data = check_cards() # This now calls your local API
    if isinstance(current_hand_data, dict) and "error" in current_hand_data:
         return {"error": "Failed to get hand for optimal pair calculation", "details": current_hand_data}

    # Placeholder for actual optimal pair logic using 'current_hand_data'
    # This logic runs on RunPod.
    print(f"RunPod: Calculating optimal pairs for hand: {current_hand_data}")
    # ... your existing logic for finding optimal pairs from a hand list ...
    # Example dummy output
    optimal_hands = [
        {"hand": current_hand_data[:5], "score": 100 * top_k, "type": "Example Pair"},
        {"hand": current_hand_data[1:6], "score": 90 * top_k, "type": "Another Example Pair"}
    ]
    return {"optimal_hands": optimal_hands[:top_k]}


def discard_cards(indices: list[int]):
    """Discard the specified card positions by calling the local game API."""
    print(f"RunPod tool: discard_cards({indices}) -> calling local game API to discard")
    result = _call_local_game_api("/game/discard", method="POST", data={"indices": indices})
    return result.get("new_hand", {"error": "No new_hand data from local API", "details": result})


def keep_cards(indices: list[int]):
    """Lock exactly five cards by calling the local game API and finish hand."""
    print(f"RunPod tool: keep_cards({indices}) -> calling local game API to keep")
    result = _call_local_game_api("/game/keep", method="POST", data={"indices": indices})
    return result # This should return score, message, etc. from local API
