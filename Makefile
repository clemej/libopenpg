all: test

test:
	nosetests --with-coverage --cover-package=. test_openpg.py
