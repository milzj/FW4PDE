pandoc --filter pandoc-citeproc --bibliography=lit_tidalfarm.bib  -f latex -t gfm -o lit.md lit.tex
