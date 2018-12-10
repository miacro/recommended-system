SHELL=/bin/bash
MAKE=make --no-print-directory

install:
	python setup.py install --user

test:
	python -m unittest discover ./recommendedsystem/test

uninstall:
	pip uninstall recommendedsystem

download-netflix-prize-dataset-torrent:
	curl -C - -fSL -o nf_prize_dataset.tar.gz.torrent http://academictorrents.com/download/9b13183dc4d60676b773c9e2cd6de5e5542cee9a.torrent

.PHONY:
	install uninstall
