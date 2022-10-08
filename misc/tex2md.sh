pandoc -s -V biblio-files=lit.bib --citeproc -f latex -t gfm -o lit.md lit.tex

