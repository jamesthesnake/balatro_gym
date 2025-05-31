#!/bin/bash

# Balatro Mac Modding Setup Script
# This script automates the installation of Lovely, Steamodded, and mods for Balatro on macOS

set -e  # Exit on any error

echo "🎮 Balatro Mac Modding Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS only!"
    exit 1
fi

# Check for required tools
print_status "Checking for required tools..."

if ! command -v curl &> /dev/null; then
    print_error "curl is required but not installed."
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    print_error "unzip is required but not installed."
    exit 1
fi

print_success "Required tools found"

# Find Steam installation
print_status "Looking for Steam and Balatro installation..."

STEAM_PATH=""
BALATRO_PATH=""

# Common Steam locations on Mac
if [ -d "$HOME/Library/Application Support/Steam" ]; then
    STEAM_PATH="$HOME/Library/Application Support/Steam"
elif [ -d "/Applications/Steam.app" ]; then
    STEAM_PATH="/Applications/Steam.app/Contents/MacOS"
fi

if [ -n "$STEAM_PATH" ]; then
    # Look for Balatro in Steam
    if [ -d "$STEAM_PATH/steamapps/common/Balatro" ]; then
        BALATRO_PATH="$STEAM_PATH/steamapps/common/Balatro"
    fi
fi

# If not found, ask user
if [ -z "$BALATRO_PATH" ]; then
    print_warning "Could not automatically find Balatro installation."
    echo "Please drag your Balatro folder into this terminal window, or type the full path:"
    read -r BALATRO_PATH
    BALATRO_PATH=$(echo "$BALATRO_PATH" | sed 's/^[ \t]*//;s/[ \t]*$//' | sed "s/[\'\"]//g")
fi

if [ ! -d "$BALATRO_PATH" ]; then
    print_error "Balatro path does not exist: $BALATRO_PATH"
    exit 1
fi

print_success "Found Balatro at: $BALATRO_PATH"

# Set up directories
BALATRO_APPDATA="$HOME/Library/Application Support/Balatro"
MODS_DIR="$BALATRO_APPDATA/Mods"
TEMP_DIR="/tmp/balatro_setup"

print_status "Creating directories..."
mkdir -p "$BALATRO_APPDATA"
mkdir -p "$MODS_DIR"
mkdir -p "$TEMP_DIR"

print_success "Directories created"

# Download and install Lovely Injector
print_status "Downloading Lovely Injector..."

# Detect Mac architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    LOVELY_ARCH="aarch64-apple-darwin"
else
    LOVELY_ARCH="x86_64-apple-darwin"
fi

print_status "Detected architecture: $ARCH (using $LOVELY_ARCH)"

# Get the download URL for Mac
LOVELY_URL=$(curl -s https://api.github.com/repos/ethangreen-dev/lovely-injector/releases/latest | grep "browser_download_url.*$LOVELY_ARCH.*tar.gz" | cut -d '"' -f 4)

if [ -z "$LOVELY_URL" ]; then
    print_error "Could not find Mac version of Lovely Injector for architecture $LOVELY_ARCH"
    exit 1
fi

cd "$TEMP_DIR"
curl -L -o lovely.tar.gz "$LOVELY_URL"
tar -xzf lovely.tar.gz

# Find the required Mac files
LOVELY_DYLIB=$(find . -name "liblovely.dylib" -type f | head -1)
LOVELY_SCRIPT=$(find . -name "run_lovely_macos.sh" -type f | head -1)

if [ -z "$LOVELY_DYLIB" ] || [ -z "$LOVELY_SCRIPT" ]; then
    print_error "Could not find liblovely.dylib or run_lovely_macos.sh in Lovely package"
    exit 1
fi

cp "$LOVELY_DYLIB" "$BALATRO_PATH/"
cp "$LOVELY_SCRIPT" "$BALATRO_PATH/"
chmod +x "$BALATRO_PATH/run_lovely_macos.sh"
print_success "Lovely Injector installed"

# Download and install Steamodded
print_status "Downloading Steamodded..."

curl -L -o steamodded.zip "https://github.com/Steamodded/smods/archive/refs/heads/main.zip"
unzip -o steamodded.zip

# Move Steamodded to mods directory
if [ -d "smods-main" ]; then
    cp -r smods-main "$MODS_DIR/"
    print_success "Steamodded installed"
else
    print_error "Could not find smods-main directory"
    exit 1
fi

# Download and install remotro mod
print_status "Downloading remotro mod..."

# Try to download the remotro mod
if curl -L -o remotro.zip "https://github.com/remotro/mod/archive/refs/heads/main.zip" 2>/dev/null; then
    if unzip -o remotro.zip 2>/dev/null; then
        # Find the mod directory
        REMOTRO_DIR=$(find . -name "mod-main" -type d | head -1)
        if [ -n "$REMOTRO_DIR" ]; then
            cp -r "$REMOTRO_DIR" "$MODS_DIR/remotro"
            print_success "Remotro mod installed"
        else
            print_warning "Could not find remotro mod files - you may need to install it manually"
        fi
    else
        print_warning "Could not extract remotro mod - you may need to install it manually"
    fi
else
    print_warning "Could not download remotro mod - you may need to install it manually"
fi

# Set permissions
print_status "Setting permissions..."
chmod -R 755 "$MODS_DIR"
chmod 755 "$BALATRO_PATH/liblovely.dylib"
chmod +x "$BALATRO_PATH/run_lovely_macos.sh"

print_success "Permissions set"

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Installation Complete!"
echo "======================="
echo ""
echo "📁 Balatro Path: $BALATRO_PATH"
echo "📁 Mods Directory: $MODS_DIR"
echo ""
echo "⚠️  IMPORTANT MAC-SPECIFIC STEPS:"
echo "1. macOS may show security warnings for liblovely.dylib"
echo "2. Go to System Preferences → Security & Privacy → General"
echo "3. Click 'Allow Anyway' for liblovely.dylib"
echo "4. You MUST launch Balatro using the run_lovely_macos.sh script, not directly!"
echo ""
echo "🚀 How to Launch Balatro with Mods:"
echo "1. Open Terminal"
echo "2. Navigate to: $BALATRO_PATH"
echo "3. Run: ./run_lovely_macos.sh"
echo "   OR create an alias: alias balatro='cd \"$BALATRO_PATH\" && ./run_lovely_macos.sh'"
echo ""
echo "🚀 Next Steps:"
echo "1. Launch Balatro using: cd \"$BALATRO_PATH\" && ./run_lovely_macos.sh"
echo "2. Look for a 'Mods' button on the main menu"
echo "3. Enable the mods you want to use"
echo "4. A debug window should appear alongside the game"
echo ""
echo "📚 Mod Locations:"
echo "- Steamodded: $MODS_DIR/smods-main"
echo "- Remotro: $MODS_DIR/remotro"
echo ""
echo "🔧 Troubleshooting:"
echo "- If macOS blocks files: System Preferences → Security & Privacy"
echo "- If permissions fail: Run 'chmod 755' on the mod folders"
echo "- MUST use run_lovely_macos.sh script to launch, not Steam directly!"
echo "- Join the Balatro Discord for mod support"
echo ""

# Ask about creating launch alias
read -p "Would you like to create a Terminal alias 'balatro' to easily launch the game? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SHELL_RC=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_RC="$HOME/.bash_profile"
    fi
    
    if [ -n "$SHELL_RC" ]; then
        echo "alias balatro='cd \"$BALATRO_PATH\" && ./run_lovely_macos.sh'" >> "$SHELL_RC"
        print_success "Alias added to $SHELL_RC"
        echo "Restart your terminal or run 'source $SHELL_RC', then type 'balatro' to launch!"
    else
        print_warning "Could not detect shell type. You can manually add this alias:"
        echo "alias balatro='cd \"$BALATRO_PATH\" && ./run_lovely_macos.sh'"
    fi
fi
echo ""

# Check if we should open directories
read -p "Would you like to open the mods directory in Finder? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open "$MODS_DIR"
fi

print_success "Setup complete! Happy modding! 🎮"
