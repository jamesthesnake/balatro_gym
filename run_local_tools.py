# local_api_server_for_live_game.py (on your Laptop)
import asyncio
from fastapi import FastAPI, HTTPException, Header # Using FastAPI for async
import uvicorn
import os
from typing import List, Dict, Any

# --- Assuming the balatro library from Smart Analyzer is accessible ---
# You'll need to structure your project so these imports work.
# This might mean the `balatro` library is installed or its path is added.
from balatro import Balatro, ScreenType  # Core library
from balatro.play import Play, HandCard  # For Play screen interactions
from balatro.deck import Card as BalatroCard, Rank, Suit # Game's Card object
# Import helper functions from your Smart Analyzer, or redefine them here
# e.g., rank_to_value, evaluate_hand, find_all_combinations_ranked

# --- Card Conversion for LLM ---
# (Similar to what we discussed before, to convert BalatroCard to "8S" format)
RANK_TO_CODE = {
    Rank.Two: '2', Rank.Three: '3', Rank.Four: '4', Rank.Five: '5',
    Rank.Six: '6', Rank.Seven: '7', Rank.Eight: '8', Rank.Nine: '9',
    Rank.Ten: 'T', Rank.Jack: 'J', Rank.Queen: 'Q', Rank.King: 'K', Rank.Ace: 'A'
}
SUIT_TO_CODE = {
    Suit.Spades: 'S', Suit.Clubs: 'C', Suit.Hearts: 'H', Suit.Diamonds: 'D'
}

def format_balatro_card_to_code(card: BalatroCard) -> str:
    if not card: return ""
    return RANK_TO_CODE.get(card.rank, '?') + SUIT_TO_CODE.get(card.suit, '?')

def format_handcards_to_codes(hand_cards: List[HandCard]) -> List[str]:
    return [format_balatro_card_to_code(hc.card) for hc in hand_cards]

# --- API Key ---
EXPECTED_API_KEY = os.environ.get("LOCAL_GAME_API_KEY", "your-secret-key-for-local-game")

async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

app = FastAPI(dependencies=[Depends(verify_api_key)])

# --- Global Balatro Game Connection State ---
# This is tricky. The SmartAnalyzer waits for a connection.
# For an API, we need a way to access an *active* Balatro object.
# This might involve running the SmartAnalyzer's connection listener
# in a background task and having it update a global `current_balatro_game_client`.
# For now, let's assume `get_active_balatro_client()` can provide this.
# This is the most complex part to integrate cleanly with FastAPI.

# Placeholder for how you'd get the active, connected Balatro instance.
# This would be linked to the `SmartAnalyzer`'s connection handling logic.
# You would need to adapt SmartAnalyzer to store the 'balatro' instance
# globally or pass it to these API handlers.
# For this example, we'll assume it's magically available.
# A real implementation would run SmartAnalyzer's core connection loop
# and allow FastAPI to access the resulting 'balatro' object.

async def get_active_balatro_client() -> Balatro:
    # In a real setup:
    # 1. Your FastAPI app would also manage an asyncio task that runs
    #    the SmartAnalyzer's server part (await asyncio.start_server(...)).
    # 2. When a game connects, the `handle_connection` from SmartAnalyzer
    #    would get a `balatro` object. This object needs to be stored globally
    #    or made accessible to your API route handlers.
    # This is a simplified placeholder.
    if not hasattr(app.state, "balatro_client") or app.state.balatro_client is None:
        raise HTTPException(status_code=503, detail="Balatro game not connected to local API server.")
    return app.state.balatro_client

# --- API Endpoints ---
@app.get("/game/state")
async def get_game_state():
    balatro = await get_active_balatro_client()
    screen = await balatro.screen() # Get current screen from live game

    if screen.type == ScreenType.PLAY:
        play_screen: Play = screen.play
        hand_cards = play_screen.hand()
        state_info = {
            "screen_type": screen.type.name,
            "hand": format_handcards_to_codes(hand_cards),
            "score": play_screen.score(),
            "money": play_screen.money(),
            "hands_left": play_screen.hands(),
            "discards_left": play_screen.discards(),
            "blind_info": play_screen.blind(), # This is usually a dict
            "selected_cards_indices": play_screen.selected_idx() # Indices of currently clicked cards
        }
        return state_info
    elif screen.type == ScreenType.SELECT_BLIND:
        # ... extract info from screen.select_blind ...
        return {"screen_type": screen.type.name, "blinds": screen.select_blind.blinds_info()}
    # Add handlers for other screen types if needed (MENU, SHOP, GAME_OVER)
    else:
        return {"screen_type": screen.type.name, "message": "Unhandled screen type for detailed state."}

@app.post("/game/action/click_cards")
async def click_cards_in_game(payload: Dict[str, List[int]]):
    balatro = await get_active_balatro_client()
    indices = payload.get("indices")
    if indices is None:
        raise HTTPException(status_code=400, detail="Missing 'indices' in payload")

    screen = await balatro.screen()
    if screen.type != ScreenType.PLAY:
        raise HTTPException(status_code=400, detail=f"Action only valid in PLAY screen, current: {screen.type.name}")

    play_screen: Play = screen.play
    try:
        # The balatro library's play.click() might return the updated Play object
        updated_play_screen = await play_screen.click(indices)
        # Or it might modify play_screen in place and return something else.
        # Assuming it returns the updated Play screen object or we refetch.
        if hasattr(updated_play_screen, 'hand'): # Check if it's a Play object
             current_selection = updated_play_screen.selected_idx()
        else: # Refetch screen if click doesn't return the Play object directly
             current_screen_after_click = await balatro.screen()
             if current_screen_after_click.type == ScreenType.PLAY:
                  current_selection = current_screen_after_click.play.selected_idx()
             else:
                  current_selection = []

        return {"message": "Cards clicked/unclicked.", "selected_indices": current_selection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clicking cards: {str(e)}")


@app.post("/game/action/discard")
async def discard_in_game(): # Assumes cards to discard are already clicked via /click_cards
    balatro = await get_active_balatro_client()
    screen = await balatro.screen()
    if screen.type != ScreenType.PLAY:
        raise HTTPException(status_code=400, detail=f"Action only valid in PLAY screen, current: {screen.type.name}")

    play_screen: Play = screen.play
    if play_screen.discards() <= 0:
        raise HTTPException(status_code=400, detail="No discards left.")

    try:
        discard_result = await play_screen.discard() # This returns a DiscardResult object
        # After discard, game state changes, fetch new state
        new_screen = await balatro.screen()
        new_state = {}
        if new_screen.type == ScreenType.PLAY:
             new_state = {
                "screen_type": new_screen.type.name,
                "hand": format_handcards_to_codes(new_screen.play.hand()),
                "discards_left": new_screen.play.discards(),
                # ... other relevant fields
             }
        elif new_screen.type == ScreenType.GAME_OVER: # Or other types
             new_state = {"screen_type": new_screen.type.name}

        return {"discard_action_result": discard_result.type.name, "new_game_state": new_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during discard: {str(e)}")


@app.post("/game/action/play_hand")
async def play_hand_in_game(): # Assumes cards to play are already clicked via /click_cards
    balatro = await get_active_balatro_client()
    screen = await balatro.screen()
    if screen.type != ScreenType.PLAY:
        raise HTTPException(status_code=400, detail=f"Action only valid in PLAY screen, current: {screen.type.name}")

    play_screen: Play = screen.play
    if play_screen.hands() <= 0:
        raise HTTPException(status_code=400, detail="No hands left to play.")
    if len(play_screen.selected_idx()) == 0: # Or not 5, depending on game rules for playing
        raise HTTPException(status_code=400, detail="No cards selected to play.")

    try:
        play_result = await play_screen.play() # This returns a PlayResult object
        # After play, game state changes significantly (new screen, round over, game over)
        new_screen_after_play = await balatro.screen()
        response_detail = {
            "play_action_result": play_result.type.name,
            "new_screen_type": new_screen_after_play.type.name
        }
        # Add more details based on play_result and new_screen_after_play
        # e.g. if play_result.type == PlayResult.ROUND_OVER, include overview.total_earned()
        # if play_result.type == PlayResult.AGAIN, include new hand from new_screen_after_play.play.hand()

        return response_detail
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during play: {str(e)}")


@app.get("/game/optimal_plays") # Example, could take top_k as query param
async def get_ranked_plays():
    balatro = await get_active_balatro_client()
    screen = await balatro.screen()
    if screen.type != ScreenType.PLAY:
        raise HTTPException(status_code=400, detail=f"Action only valid in PLAY screen, current: {screen.type.name}")

    play_screen: Play = screen.play
    hand_cards: List[HandCard] = play_screen.hand()

    # Use the find_all_combinations_ranked from your Smart Analyzer script
    # Ensure it's adapted to take List[HandCard] and returns serializable data.
    # ranked_combinations = find_all_combinations_ranked(hand_cards)
    # For this example, a placeholder:
    ranked_combinations_placeholder = [
        {"score": 100, "indices": [0,1,2,3,4], "hand_name": "Example High Hand", "cards": format_handcards_to_codes(hand_cards[:5])},
        {"score": 50, "indices": [1,2,3,4,5], "hand_name": "Example Mid Hand", "cards": format_handcards_to_codes(hand_cards[1:6] if len(hand_cards) > 5 else [])}
    ]
    return {"optimal_plays": ranked_combinations_placeholder} # ranked_combinations

# --- Main section to run API and integrate SmartAnalyzer's connection ---
# This part needs careful integration of SmartAnalyzer's asyncio server
# with FastAPI's Uvicorn server. One way is to run SmartAnalyzer's
# connection logic in a background asyncio task.

async def run_balatro_connector(app_state):
    # Simplified from SmartAnalyzer.run and SmartAnalyzer.handle_connection
    # This should establish the connection and store the 'balatro' client object
    # in app_state for API routes to use. This is non-trivial.
    # This is where you'd use HOST and PORT from SmartAnalyzer.
    # For now, this part is conceptual and needs the actual connection logic.
    logger.info("Background Balatro connector started (conceptual).")
    # server = await asyncio.start_server(lambda r,w: handle_game_conn(r,w,app_state), HOST, PORT)
    # async with server: await server.serve_forever()
    # Actual implementation would involve `Connection(reader,writer)` and `Balatro(connection)`
    # and storing the `balatro` object in `app_state.balatro_client`.
    pass # Replace with actual connection logic.

@app.on_event("startup")
async def startup_event():
    # If you want to run the balatro connector in the background
    # asyncio.create_task(run_balatro_connector(app.state))
    # For now, let's assume the game connects and the balatro_client is set externally or manually for testing
    app.state.balatro_client = None # Initialize; needs to be set by actual game connection
    print("Local API Server started. Waiting for game to connect and set balatro_client.")
    print(f"Ensure LOCAL_GAME_API_KEY is set. Use X-API-Key header for auth.")
    print(f"Ngrok should point to this server's host/port (e.g., localhost:8001 if running locally).")


# To run (example): uvicorn local_api_server_for_live_game:app --host 0.0.0.0 --port 8001
# You'd also need to start your ngrok tunnel: ngrok http 8001
