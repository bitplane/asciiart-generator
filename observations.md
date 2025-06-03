# Glyph Behavior Observations

## Width Categories
- **0-width glyphs**: None appear to be printable (expected combining characters, but none found)
- **1-width glyphs**: 6,766 found (standard monospace)
- **2-width glyphs**: 11,760 found (e.g., UnJamaNovel font)

## Glyph Properties

### Box Drawing Characters
Example set: `━│┃┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋▒■`

These can be analyzed by:
- **Continuity**: Connection points with neighboring cells
- **Compatibility**: Which glyphs flow well together
- **Directionality**: Which edges extend outward
- **Density**: Amount of "ink" in the glyph
- **Center of mass**: Weight distribution
- **Noise**: Texture/pattern elements

### Bleeding/Overflow Behavior

Some glyphs extend beyond their designated cell boundaries, creating overlapping effects.

**Test case:**
```bash
printf "%s⒬%s⒭%s\n" $'\033[37;40m' $'\033[30;47m' $'\033[0m'
```
Result: Shows 3 colors in the middle of the second glyph due to bleeding

**With invert attribute:** The bleeding gets truncated

## Attribute Interactions

### Bleeding Behavior Rules
1. **Invert attribute**: STOPS bleeding (truncates the overflow)
2. **Background color changes**: Do NOT stop bleeding
3. **Foreground/background swap optimization**: When fg/bg colors are swapped, terminal optimizes to invert, which then stops bleeding

### Terminal Optimization Discovery
The terminal appears to detect when:
```
foreground=A, background=B  →  foreground=B, background=A
```
And optimizes this to use the invert attribute instead, which has the side effect of stopping bleed.

This suggests bleeding is handled at a lower level than color attributes but is affected by text rendering attributes like invert.

Tested interactions:
- ✅ Invert: truncates bleeding
- ✅ Background color: bleeding continues  
- ✅ Foreground color: bleeding continues
- ✅ FG/BG swap: optimized to invert, stops bleeding
- ✅ Strikethrough: stays within cell (doesn't affect bleeding)
- ✅ Underline: stays within cell (doesn't affect bleeding)
- ❓ Bold: needs testing

## Combinatorial Space

With 6,766 single-width glyphs:
- **Pairs**: 45,778,756 possible combinations
- **Plus**: 11,760 double-width glyphs
- **Context**: Each position has 8 neighboring regions
- **Attributes**: Multiple attribute combinations multiply the space further

## Research Questions

1. How to detect bleeding programmatically?
   - Use PIL to render glyphs and check pixel bounds?
   - Compare rendered area vs expected cell boundaries?

2. How do different terminals handle bleeding?
   - Test across terminal emulators
   - Document consistency/inconsistencies

3. Can bleeding be exploited for effects?
   - Smooth transitions between cells
   - Anti-aliasing effects
   - Artistic overlaps

## Hierarchical Block Rendering Idea

Starting with a full block (██) representing dominant colors, we can create multi-scale representations:

### Level 1: Simple block
```
██
```

### Level 2: Internal structure with quarter-character alignment
```
AAAABBBB
AA╔══╗BB  
CC╚══╝DD
CCCCDDDD
```

Where A, B, C, D are overlapping color/pattern groups that create "jigsaw" constraints between blocks.

### Key Insights:
- Quarter-block overlapping groups ensure smooth transitions
- Box drawing characters represent internal structure
- Edge constraints maintain continuity between neighboring blocks
- Multi-scale: blocks → box drawing → braille → bleeding chars
- Creates a constraint-satisfaction problem for coherent rendering

### Character vocabulary:
- **Blocks**: █▀▄▌▐░▒▓ (dominant regions)
- **Box drawing**: ┌┐└┘├┤┬┴┼─│╔╗╚╝╠╣╦╩╬═║ (structure)
- **Braille**: ⠀⠁⠂⠃...⠿ (2x4 dot matrix = 256 patterns)
- **Bleeding**: ⒜⒝⒞... (smooth transitions)

This approach could create novel aesthetics where blocks "understand" their context and fit together like puzzle pieces.

## Next Steps

1. Create a script to systematically test glyph bleeding using PIL
2. Map which glyphs bleed and in which directions
3. Test attribute effects on bleeding glyphs
4. Build a database of glyph properties and relationships
5. Implement hierarchical block rendering prototype