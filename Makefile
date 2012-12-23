all: test

test:
	nosetests --with-coverage --cover-package=openpg --cover-package=isomorphic test_openpg.py
