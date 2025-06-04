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

### Font Fallback and Bleeding
**Critical discovery**: Gnome terminal uses font fallback chains that can fall back to **non-monospace fonts** for characters not available in the primary monospace font.

**Example**: The character `⸺` (U+2E3A, two-em dash):
- **Primary font**: UbuntuSansMono (monospace) - character not available
- **Fallback font**: NotoSans-Regular (proportional) - character available
- **Result**: Character is 7.75x wider than terminal cell (31px vs 4px space width)

**Investigation Results**:
```
Font              Cell Width    Char Width    Bleeding
NotoSans-Regular      4px          31px       7.75x (massive)
FreeSans              4px          32px       8x (massive)  
DejaVuSans            5px          10px       2x (moderate)
FreeMono (mono)      10px          10px       None (fits)
```

**Implications**:
- PIL font rendering matches terminal behavior when using correct fallback font
- Same Unicode character renders completely differently across fonts
- Monospace fonts constrain characters; proportional fonts allow massive bleeding
- Bleeding detection must use font resolution to identify actual rendering font

### Performance Discovery: Font Resolution Bottleneck
**Critical insight**: Unicode fallback resolution is painfully slow, making brute-force scanning of millions of Unicode codepoints infeasible.

**Problem**: Character-by-character font resolution approach:
- Font resolution lookup per character: ~0.5-2 chars/sec
- Total Unicode space: 1.1M codepoints  
- Estimated time: 6-24 days

**Solution**: Font-first approach instead of character-first:
- Load each font once and extract its character map
- Process only characters that exist in each font
- Skip proportional fonts entirely
- Result: 7,402 quarterable glyphs processed in 5.3 seconds

**Data Efficiency**: 
- 309,809 unique quarter patterns
- Total data: 36MB raw, <10MB compressed
- **No symmetry optimization yet** - could reduce further by detecting rotations/mirrors

**Search Space Collapse**:
- Theoretical: Millions of Unicode combinations
- Actual: 310k unique quarter patterns (~96B possible pairs)
- Manageable size for compatibility analysis

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

## Quarter-Block Compatibility Graph

The quarter-block idea naturally creates a compatibility graph between characters. 

### Implementation Strategy:
1. **Group chars by** `(size: int, overlap: bool)`
2. **Break each char into quarters**
3. **Develop hash/key metrics** for quarter comparison

### Font Dimension Guidelines:
- Use "round" numbers only (multiples of 2, ideally powers of 2)
- Ensures natural stride alignment with features
- Examples: 8×8, 16×16, 8×16 character cells

### Quarter-Block Evaluation Metrics (Brainstorming):

#### Density Metrics
- **Ink density**: Percentage of pixels filled in quarter
- **Edge density**: Pixels on quarter boundaries
- **Corner density**: Pixels at quarter corners

#### Directional Metrics  
- **Horizontal flow**: ∑(pixel[x] - pixel[x-1]) 
- **Vertical flow**: ∑(pixel[y] - pixel[y-1])
- **Diagonal flow**: NE, NW, SE, SW gradients
- **Dominant direction**: Strongest flow vector

#### Structural Metrics
- **Center of mass**: (x̄, ȳ) within quarter
- **Moment of inertia**: Spread around center
- **Connectivity**: Number of connected components
- **Holes**: Number of enclosed empty regions

#### Edge Compatibility Metrics
- **Edge signature**: Binary pattern of pixels on each edge
- **Edge continuity score**: How well edges match neighbors
- **Corner types**: Empty, filled, L-shape, diagonal
- **Edge gradient**: Smooth vs sharp transitions

#### Frequency/Texture Metrics
- **DCT coefficients**: 2D frequency components
- **Local variance**: Texture roughness
- **Periodicity**: Repeating patterns
- **Entropy**: Information content

#### Perceptual Hashes
- **Gradient hash**: Quantized directional gradients
- **Block hash**: Reduced resolution comparison
- **Radial hash**: Polar coordinate features
- **Wavelet hash**: Multi-scale features

### Compatibility Scoring
```
compatibility(Q1, Q2, edge) = weighted_sum(
    edge_match_score,
    flow_continuity,
    density_similarity,
    structural_coherence
)
```

## Next Steps

1. Create a script to systematically test glyph bleeding using PIL
2. Map which glyphs bleed and in which directions
3. Test attribute effects on bleeding glyphs
4. Build a database of glyph properties and relationships
5. Implement quarter-block analysis with these metrics
6. Build compatibility graph between character quarters