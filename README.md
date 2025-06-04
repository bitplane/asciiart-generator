# ASCII art generator research project

Looking at novel ways to make an ascii art generator

Current thinking:

* Use TTY to measure how far the cursor moves, to group by width.
* Measure which ones actually stay in their zone and which ones bleed out into
  other regions.
* There are cool tricks we can do with ones that do bleed, as it allows us to
  render 3 colours in one square.
* All the 2 char ones don't bleed into other squares so we don't include them.
* Start with the 1 char non-bleeding ones and see what they look like in my
  terminal.Å -
* We break this into quarters. We use a sensible size in pixels so this is an
  even number.
* We get some metrics from each quarter
