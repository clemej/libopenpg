libopenpg
=========

A Python library implementing Open Planar Graphs

Requirements
============

This python library uses the NetworkX graph library, so you will need that
installed on your system to use the library.  On debian-derived Linux 
distributions you can install this with the command

 $ sudo apt-get install python-networkx

Overview
========

This library is a work-in-progress implementation of Open Planar Graphs
as descriped in the paper

> "Polynomial Algorithms for Open Plane Graph and Subgraph Isomorphism by 
de la Higuera, Janodet, Samuel, et al."

Defining Graphs
===============

Please see the file example\_graphs.py for how to specify graphs.  Basically,
You greate a graph with

	G = openpg.openpg()

A graph is defined through its faces, which are created through defining a list
of nodes like so:

	f1 = openpg.face([00,1,11,90,11,10], visible=True, outer=False)

And then define faces through a list of nodes.  The nodes can be anything
(strings, numbers, etc..).  You then have to add faces to the graph using
the add\_face mthod:

	G.add\_face(f1)

To make things a proper Open Plane Graph, you must of course add an outer face:

> Note: There is currently very little checking if a graph is valid before 
running the algorithms. 

Operations
==========

To see the list of operations on openpg objects, run:

	$ python
	> import openpg
	> help(openpog.openpg)

And see the examples in example.py .

The isomorphic tests are in a separate module, and you can view the public
members the same way.








