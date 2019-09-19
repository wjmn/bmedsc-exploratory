(TeX-add-style-hook
 "thesis"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("book" "12pt")))
   (TeX-run-style-hooks
    "latex2e"
    "book"
    "bk12"))
 :latex)

