import openpg

def example_graph1(name='square6'):
	"""
	Returns a simple graph:

	o---o---o---o
	| 1 | 2 | 3 |   Nodes numbered on grid from 0,0 to 3,2. left to right,  
	o---o---o---o                                           top to bottom.
	| 4 | 5 | 6 |   f1, f2, f5, f6 are visible. 
	o---o---o---o

	"""
	G = openpg.openpg(name=name)

	fouter = openpg.face([0,10,20,30,31,32,22,12,2,1], visible=False,
								outer=True)
	f1 = openpg.face([00,1,11,90,11,10], visible=True)
	f2 = openpg.face([10,11,21,20], visible=True)
	f3 = openpg.face([20,21,31,30])
	f4 = openpg.face([1,2,12,11])
	f5 = openpg.face([11,12,22,21], visible=True)
	f6 = openpg.face([21,22,32,31], visible=True)

	G.add_face(fouter)
	G.add_face(f1)
	G.add_face(f2)
	G.add_face(f3)
	G.add_face(f4)
	G.add_face(f5)
	G.add_face(f6)

	return G

def example_graph2():
	"""
	Same as graph1, but with some pendent nodes added. 
	"""
	G = openpg.openpg(name='square6-extras')

	fouter = openpg.face([00,10,20,30,31,32,22,12,2,1,70,71,70,1,60,1], 
					visible=False, outer=True)
	f1 = openpg.face([00,1,11,90,11,10], visible=True)
	f2 = openpg.face([10,11,21,20], visible=True)
	f3 = openpg.face([20,21,31,30])
	f4 = openpg.face([1,2,12,11,80,11])
	f5 = openpg.face([11,12,22,21], visible=True)
	f6 = openpg.face([21,22,32,31], visible=True)

	G.add_face(fouter)
	G.add_face(f1)
	G.add_face(f2)
	G.add_face(f3)
	G.add_face(f4)
	G.add_face(f5)
	G.add_face(f6)

	return G

def example_graph3(name=''):
	G = openpg.openpg(name=name)

	fouter = openpg.face([-1,-2,-3,-4], visible=False, outer=True)
	ftop = openpg.face([-3,-2,-1,00,10,20,30,31,32], visible=True)
	fbottom = openpg.face([00,-1,-4,-3,32,22,12,2,1], visible=True)

	f1 = openpg.face([00,1,11,90,11,10], visible=True)
	f2 = openpg.face([10,11,21,20], visible=True)
	f3 = openpg.face([20,21,31,30])
	f4 = openpg.face([1,2,12,11])
	f5 = openpg.face([11,12,22,21])
	f6 = openpg.face([21,22,32,31], visible=True)

	G.add_face(fouter)
	G.add_face(ftop)
	G.add_face(fbottom)
	G.add_face(f1)
	G.add_face(f2)
	G.add_face(f3)
	G.add_face(f4)
	G.add_face(f5)
	G.add_face(f6)

	return G

def example_graph4(name=''):
	G = openpg.openpg(name=name)

	fouter = openpg.face([0,10,20,30,31,32,22,12,2,1], visible=False,
								outer=True)
	f1 = openpg.face([00,1,11,90,11,10], visible=True)
	f2 = openpg.face([10,11,21,20], visible=True)
	f3 = openpg.face([20,21,31,30])
	f4 = openpg.face([1,2,12,11])
	f5 = openpg.face([11,12,22,21], visible=True)
	f6 = openpg.face([21,22,32,31], visible=False)

	G.add_face(fouter)
	G.add_face(f1)
	G.add_face(f2)
	G.add_face(f3)
	G.add_face(f4)
	G.add_face(f5)
	G.add_face(f6)

	return G

