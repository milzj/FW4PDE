pandoc -s -V biblio-files=lit.bib --citeproc -f latex -t gfm -o test.md lit.tex

