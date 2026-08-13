#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SESSION_NAME="cat"

echo -e "${BLUE}=== Cat Live Session Starter ===${NC}"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}tmux is not installed. Install with: sudo apt-get install tmux${NC}"
    exit 1
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo -e "${YELLOW}Session '$SESSION_NAME' already exists.${NC}"
    read -p "Do you want to kill the existing session and restart? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Killing existing session...${NC}"
        tmux kill-session -t $SESSION_NAME
    else
        echo -e "${BLUE}Attaching to existing session...${NC}"
        tmux attach-session -t $SESSION_NAME
        exit 0
    fi
fi

# Create new tmux session in background
echo -e "${BLUE}Creating tmux session '$SESSION_NAME'...${NC}"
tmux new-session -d -s $SESSION_NAME

# Create or activate virtual environment
echo -e "${BLUE}Checking virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    tmux send-keys -t $SESSION_NAME "python3 -m venv venv" C-m
    sleep 2
fi

# Activate venv
echo -e "${BLUE}Activating virtual environment...${NC}"
tmux send-keys -t $SESSION_NAME "source venv/bin/activate" C-m
sleep 1

# Install requirements
if [ -f "requirements-minimal.txt" ]; then
    echo -e "${BLUE}Installing requirements from requirements-minimal.txt...${NC}"
    tmux send-keys -t $SESSION_NAME "pip install -r requirements-minimal.txt" C-m
    sleep 3
else
    echo -e "${YELLOW}Warning: requirements-minimal.txt not found!${NC}"
fi

# Start main.py live
echo -e "${GREEN}Starting python main.py live...${NC}"
tmux send-keys -t $SESSION_NAME "python main.py live" C-m

# Wait briefly
sleep 1

# Attach to session
echo -e "${GREEN}=== Attaching to session '$SESSION_NAME' ===${NC}"
echo -e "${YELLOW}Tip: Press Ctrl+B then D to detach${NC}"
echo -e "${YELLOW}     To reattach: tmux attach -t $SESSION_NAME${NC}"
sleep 2
tmux attach-session -t $SESSION_NAME
