

docs: _html/README.html


_html/README.html: README.rst
	sphinx-build -E -C -D master_doc=README -D html_theme_options.nosidebar=True -D html_theme=alabaster -D html_theme_options.show_powered_by=False -D pygments_style=sphinx  . _html
