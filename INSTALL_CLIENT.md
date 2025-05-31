# Remotro Balatro Remote Control Setup Guide

You now have both components of the remotro system! This guide will help you set up complete remote control of Balatro games.

## Project Structure

The remotro project consists of:
- **Client Library** (`client/lib/`): Rust library for connecting to and controlling Balatro
- **Server Mod** (`mod/`): Balatro mod that hosts the control server
- **Examples** (`client/example/`): Sample implementations

## Prerequisites

### 1. Install Rust
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 2. Set Up Balatro Modding Infrastructure
You'll need the basic Balatro modding setup first:

- Install Lovely Injector
- Install Steamodded mod loader
- Set up the mods directory: `%AppData%\Balatro\Mods` (Windows) or `~/Library/Application Support/Balatro/Mods` (macOS)

## Step 2: Build the Rust Client

### 1. Navigate to the Client Directory
```bash
cd ~/Downloads/client/lib
```

### 2. Build the Library
```bash
cargo build --release
```

### 3. Test Connection (if examples are available)
```bash
cd ../example
cargo run
```

## Step 3: Understanding the Protocol

The remotro system uses a custom protocol over LÖVE's threading system:

### Message Format
- **Format**: `kind!body`
- **Heartbeat**: Automatic ping/pong to maintain connection
- **JSON Bodies**: Structured data for complex requests

### Available Actions (from play.lua analysis)
1. **click**: Select/deselect cards by index
2. **play**: Play the selected hand  
3. **discard**: Discard selected cards
4. **State queries**: Get current hand, score, money, etc.

## Step 4: Basic Usage Example

Here's how to use the remotro client once everything is set up:

```rust
use remotro::{Remotro, Balatro};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The mod should be hosting on localhost, likely port 8080
    let mut remotro = Remotro::host("localhost", 8080).await?;
    println!("Waiting for Balatro connection...");
    
    // Accept connection from Balatro
    let mut balatro = remotro.accept().await?;
    println!("Connected to Balatro!");
    
    // Get current game state
    let play_state = balatro.play().await?;
    
    // Check what's in your hand
    println!("Current hand: {:?}", play_state.hand());
    println!("Current blind: {:?}", play_state.blind());
    println!("Score: {}", play_state.score());
    println!("Money: ${}", play_state.money());
    
    // Select first three cards (indices 0, 1, 2)
    let play_state = play_state.click(&[0, 1, 2]).await?;
    
    // Play the selected hand
    match play_state.play().await? {
        PlayResult::Again(next_state) => {
            println!("Hand played successfully, continuing...");
            // Continue with next_state for more actions
        }
        PlayResult::RoundOver(round_overview) => {
            println!("Round completed!");
            // Handle round transition, shopping, etc.
        }
        PlayResult::GameOver(game_overview) => {
            println!("Game finished!");
            // Handle game end
        }
    }
    
    Ok(())
}
```

## Step 1: Install the Remotro Server Mod

### 1. Copy the Mod to Balatro
```bash
# Copy the entire mod folder to your Balatro mods directory
# macOS:
cp -r ~/Downloads/mod ~/Library/Application\ Support/Balatro/Mods/remotro

# Windows (in PowerShell):
# Copy-Item -Recurse "C:\Users\[YourUsername]\Downloads\mod" "$env:APPDATA\Balatro\Mods\remotro"
```

### 2. Verify Mod Structure
Your Balatro mods folder should now contain:
```
Mods/
├── steamodded-main/         # (Steamodded mod loader)
└── remotro/                 # (Remotro mod)
    ├── config.lua
    ├── core.lua
    ├── lovely.toml
    ├── Remotro.json
    ├── hooks/
    │   ├── blinds.lua
    │   ├── deck.lua
    │   ├── manager.lua
    │   ├── overview.lua
    │   ├── play.lua
    │   ├── screen.lua
    │   └── ... (other hook files)
    ├── net/
    │   └── client.lua
    └── vendor/
```

### 3. Launch Balatro
1. Start Balatro (the Steamodded debug window should appear)
2. Look for the "Mods" button on the main menu
3. Enable the Remotro mod if it's not already enabled
4. Watch the debug window for any loading errors

## Connection Flow

1. **Install both Lovely Injector and Steamodded** in Balatro
2. **Copy the remotro mod** to your Balatro mods folder
3. **Start Balatro** with the remotro mod enabled
4. **The mod creates a server** using LÖVE's threading system
5. **Run your Rust client** to connect to the server
6. **Real-time bidirectional communication** between client and game

## Troubleshooting

### Common Issues

1. **Mod Not Loading**:
   - Verify Lovely and Steamodded are properly installed
   - Check file permissions on the mod folder
   - Look for errors in the Steamodded debug window
   - Ensure the mod folder is named correctly (e.g., `remotro`)

2. **Connection Issues**:
   - Check that Balatro is running with the remotro mod enabled
   - Verify the mod is actually hosting a server (check debug output)
   - Try different ports if the default doesn't work
   - Check firewall settings

3. **Build Errors**:
   - Update Rust: `rustup update`
   - Check dependencies in `Cargo.toml`
   - Ensure you're in the correct directory (`client/lib/`)

### Debug Tips

1. **Monitor the Steamodded debug window** for mod errors and connection status

2. **Enable logging** in your Rust client:
   ```rust
   env_logger::init();
   ```

3. **Check network activity** to verify connection attempts

4. **Start simple**: Begin with basic connection and state reading before attempting complex automation

## Next Steps

1. **Complete the basic setup** following the steps above
2. **Test the connection** with a simple client
3. **Explore the API** by examining available methods
4. **Build your automation** based on your specific use case

## Example Use Cases

- **AI Bot**: Fully automated Balatro gameplay with strategic decision making
- **Stream Integration**: Let viewers control game decisions via chat commands  
- **Analysis Tools**: Real-time game state logging and statistical analysis
- **Speedrun Assistance**: Optimal play suggestions and move validation
- **Remote Play**: Control Balatro from another device or over the network
- **Educational Tools**: Step-by-step game tutorials and guided play

The remotro system provides a comprehensive foundation for any programmatic Balatro interaction!
