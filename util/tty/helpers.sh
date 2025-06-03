#!/bin/bash
# Terminal testing helpers - source this file
# Usage: source test_helpers.sh

# ANSI Reset
export RESET=$'\033[0m'

# Basic colors
export BLACK=$'\033[30m'
export RED=$'\033[31m'
export GREEN=$'\033[32m'
export YELLOW=$'\033[33m'
export BLUE=$'\033[34m'
export MAGENTA=$'\033[35m'
export CYAN=$'\033[36m'
export WHITE=$'\033[37m'

# Background colors
export BG_BLACK=$'\033[40m'
export BG_RED=$'\033[41m'
export BG_GREEN=$'\033[42m'
export BG_YELLOW=$'\033[43m'
export BG_BLUE=$'\033[44m'
export BG_MAGENTA=$'\033[45m'
export BG_CYAN=$'\033[46m'
export BG_WHITE=$'\033[47m'

# Attributes
export BOLD=$'\033[1m'
export DIM=$'\033[2m'
export ITALIC=$'\033[3m'
export UNDERLINE=$'\033[4m'
export BLINK=$'\033[5m'
export REVERSE=$'\033[7m'
export STRIKETHROUGH=$'\033[9m'

# 256 color shortcuts
c256() { echo -n $'\033[38;5;'"$1"'m'; }
bg256() { echo -n $'\033[48;5;'"$1"'m'; }

# True color
rgb() { echo -n $'\033[38;2;'"$1"';'"$2"';'"$3"'m'; }
bg_rgb() { echo -n $'\033[48;2;'"$1"';'"$2"';'"$3"'m'; }

# Cursor movement
move_to() { printf '\033[%d;%dH' "$1" "$2"; }  # move_to row col
move_home() { printf '\033[H'; }               # move to 1,1
move_start() { printf '\r'; }                  # move to start of line
move_up() { printf '\033[%dA' "${1:-1}"; }     # move up N lines
move_down() { printf '\033[%dB' "${1:-1}"; }   # move down N lines
move_left() { printf '\033[%dD' "${1:-1}"; }   # move left N chars
move_right() { printf '\033[%dC' "${1:-1}"; }  # move right N chars

# Combining characters for testing
export COMBINING_ACUTE=$'\u0301'      # ́  combining acute accent
export COMBINING_GRAVE=$'\u0300'      # ̀  combining grave accent
export COMBINING_CIRCUMFLEX=$'\u0302' # ̂  combining circumflex
export COMBINING_TILDE=$'\u0303'      # ̃  combining tilde
export COMBINING_MACRON=$'\u0304'     # ̄  combining macron
export COMBINING_OVERLINE=$'\u0305'   # ̅  combining overline
export COMBINING_BREVE=$'\u0306'      # ̆  combining breve
export COMBINING_DOT=$'\u0307'        # ̇  combining dot above
export COMBINING_DIAERESIS=$'\u0308'  # ̈  combining diaeresis
export COMBINING_RING=$'\u030A'       # ̊  combining ring above

# Interesting glyphs for testing
export BOX_LIGHT="┌┐└┘├┤┬┴┼─│"
export BOX_HEAVY="┏┓┗┛┣┫┳┻╋━┃"
export BOX_DOUBLE="╔╗╚╝╠╣╦╩╬═║"
export BLOCKS="▀▄█▌▐░▒▓"
export TRIANGLES="▲▼◀▶◢◣◤◥"
export CIRCLES="●○◉◯⬤⭕"
export SHADES="░▒▓█"
export DOTS="⋅·•‧∙⦁⦿"

# Potentially bleeding glyphs
export BLEEDING_CANDIDATES="⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵"
export WIDE_CHARS="🌟⭐✨🎯🎨🎭🎪🎫🎬🎮"

# Test functions
test_bleeding() {
    local char="$1"
    echo "Testing bleeding for: $char"
    printf "%s${char}%s${char}%s${char}%s\n" "$WHITE$BG_BLACK" "$BLACK$BG_RED" "$RED$BG_BLUE" "$RESET"
    printf "%s${char}%s${char}%s${char}%s\n" "$REVERSE$WHITE$BG_BLACK" "$REVERSE$BLACK$BG_WHITE" "$REVERSE$RED$BG_BLUE" "$RESET"
}

test_attributes() {
    local char="$1"
    echo "Testing attributes for: $char"
    printf "Normal: %s\n" "$char"
    printf "Bold: %s%s%s\n" "$BOLD" "$char" "$RESET"
    printf "Underline: %s%s%s\n" "$UNDERLINE" "$char" "$RESET"
    printf "Reverse: %s%s%s\n" "$REVERSE" "$char" "$RESET"
    printf "Strikethrough: %s%s%s\n" "$STRIKETHROUGH" "$char" "$RESET"
    printf "Bold+Underline: %s%s%s%s\n" "$BOLD$UNDERLINE" "$char" "$RESET"
}

test_width() {
    local char="$1"
    echo -n "Width test for $char: "
    printf "\r%s" "$char"
    printf "\033[6n"
    read -sdR pos
    pos=${pos#*[}
    col=${pos#*;}
    col=${col%R}
    printf "\rWidth: %d     \n" $((col - 1))
}

# Quick bleeding test with your example
# Test a sequence of characters
test_sequence() {
    local chars="$1"
    echo "Testing sequence: $chars"
    for (( i=0; i<${#chars}; i++ )); do
        char="${chars:$i:1}"
        printf "%s%s" "$(c256 $((i % 256)))" "$char"
    done
    printf "%s\n" "$RESET"
}

# Show all box drawing characters
show_boxes() {
    echo "Light box drawing:"
    echo "$BOX_LIGHT"
    echo "Heavy box drawing:"
    echo "$BOX_HEAVY"
    echo "Double box drawing:"
    echo "$BOX_DOUBLE"
}

# Show color palette
show_palette() {
    echo "16 color palette:"
    for i in {0..15}; do
        printf "%s%3d%s " "$(c256 $i)" "$i" "$RESET"
        [[ $((i % 8)) == 7 ]] && echo
    done
    echo
}

# Show available test functions
# Test combining characters
test_combining() {
    local base="$1"
    echo "Testing combining chars with: $base"
    printf "Base: %s\n" "$base"
    printf "Acute: %s%s\n" "$base" "$COMBINING_ACUTE"
    printf "Grave: %s%s\n" "$base" "$COMBINING_GRAVE"
    printf "Circumflex: %s%s\n" "$base" "$COMBINING_CIRCUMFLEX"
    printf "Tilde: %s%s\n" "$base" "$COMBINING_TILDE"
    printf "Multiple: %s%s%s%s\n" "$base" "$COMBINING_ACUTE" "$COMBINING_DOT" "$COMBINING_RING"
}

show_help() {
    echo "Available functions:"
    echo "  test_bleeding <char>     - Test if character bleeds"
    echo "  test_attributes <char>   - Test character with different attributes"
    echo "  test_width <char>        - Measure character width"
    echo "  test_combining <char>    - Test combining characters"
    echo "  quick_bleed_test         - Run your ⒬⒭ example"
    echo "  test_sequence <string>   - Test a string with rainbow colors"
    echo "  show_boxes              - Display box drawing characters"
    echo "  show_palette            - Show color palette"
    echo ""
    echo "Colors:"
    echo "  c256 <num>              - Set 256 color"
    echo "  bg256 <num>             - Set 256 background color"
    echo "  rgb <r> <g> <b>         - Set RGB color"
    echo "  bg_rgb <r> <g> <b>      - Set RGB background color"
    echo ""
    echo "Movement:"
    echo "  move_to <row> <col>     - Move cursor to position"
    echo "  move_home               - Move to 1,1"
    echo "  move_start              - Move to start of line"
    echo "  move_up/down/left/right [N] - Move cursor"
    echo ""
    echo "Character sets:"
    echo "  \$BOX_LIGHT, \$BOX_HEAVY, \$BOX_DOUBLE"
    echo "  \$BLOCKS, \$TRIANGLES, \$CIRCLES, \$SHADES, \$DOTS"
    echo "  \$BLEEDING_CANDIDATES, \$WIDE_CHARS"
    echo "  \$COMBINING_* (ACUTE, GRAVE, CIRCUMFLEX, etc.)"
}

echo "Terminal test helpers loaded! Type 'show_help' for available functions."
