(TeX-add-style-hook
 "thesis3"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem") ("geometry" "margin=1.4in") ("natbib" "sort&compress" "numbers") ("caption" "font=small" "labelfont=bf")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref"
    "helvet"
    "gensymb"
    "xcolor"
    "tikz"
    "microtype"
    "tabularx"
    "tabu"
    "geometry"
    "natbib"
    "caption")
   (LaTeX-add-labels
    "sec:orgba2b151"
    "sec:org504c71e"
    "sec:org26abd49"
    "sec:org5522edd"
    "sec:org0bb5ba4"
    "sec:orgf70642c"
    "sec:org723690a"
    "sec:org989ea9d"
    "sec:orgca20e25"
    "sec:orgfaf8140"
    "sec:org8efb8ed"
    "sec:org1c1a8dd"
    "sec:orga7551ed"
    "sec:org964d9b1"
    "sec:orged50db3")
   (LaTeX-add-bibliographies
    "refs"))
 :latex)

