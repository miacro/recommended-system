SHELL=/bin/bash
MAKE=make --no-print-directory

install:
	python setup.py install --user

test:
	python -m unittest discover ./recommendedsystem/test

uninstall:
	pip uninstall recommendedsystem

.PHONY:
	install uninstall
