(TeX-add-style-hook
 "thesis3"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("book" "11pt")))
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
    "book"
    "bk11"
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
    "appendix"
    "tikz"
    "microtype"
    "tabularx"
    "tabu"
    "geometry"
    "natbib"
    "caption")
   (LaTeX-add-labels
    "sec:org8ef7dd2"
    "sec:org00ab7df"
    "sec:orgca01afd"
    "sec:org4568c3f"
    "sec:orgde30108"
    "sec:org0e1c73b"
    "sec:org6b13db8"
    "sec:org08f06b5"
    "sec:org82d09ce"
    "sec:org60bfeb3"
    "sec:org0aa64c8"
    "fig:org3e45fac"
    "sec:org79c4924"
    "fig:org593149f"
    "sec:org617e7cf"
    "sec:org9597c77"
    "sec:org0633bed"
    "sec:org674f96a"
    "sec:org8626dd4"
    "sec:orge7eca6f"
    "sec:org5606157"
    "sec:orgc01aa4c"
    "sec:orgd15dd4b"
    "sec:org413193f"
    "sec:org5c2b0a3"
    "sec:org78ce417"
    "sec:org6ad2c6f"
    "sec:org51e4b6a"
    "sec:orge0af37b"
    "sec:org3685ede"
    "sec:org5e02e33"
    "sec:org98890eb"
    "sec:org588aeae"
    "sec:org0144534"
    "sec:orga1c123e"
    "sec:org5e53f6c"
    "sec:org5d09f8e"
    "sec:org90ac40d"
    "sec:orgae492dd"
    "sec:orge7386e2"
    "sec:org7a22174"
    "sec:org98cb18e"
    "sec:orgc1fd62b"
    "sec:org059508e"
    "sec:org521acb7"
    "sec:orgdc9136a"
    "sec:org6eab54b"
    "sec:org91c8641"
    "sec:orga39179b"
    "sec:orgf041404"
    "sec:orgfa043aa"
    "sec:org8c98f39"
    "sec:orge2fe194")
   (LaTeX-add-bibliographies
    "refs"))
 :latex)

