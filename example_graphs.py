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

	fouter = openpg.face([00,10,20,30,31,32,22,12,02,01], visible=False,
								outer=True)
	f1 = openpg.face([00,01,11,90,11,10], visible=True)
	f2 = openpg.face([10,11,21,20], visible=True)
	f3 = openpg.face([20,21,31,30])
	f4 = openpg.face([01,02,12,11])
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
        G = example_graph1(name='square6-extras')

	fouter = openpg.face([00,10,20,30,31,32,22,12,02,01,70,71,70,01,60,01], 
					visible=False, outer=True)
	f1 = openpg.face([00,01,11,90,11,10], visible=True)
	f2 = openpg.face([10,11,21,20], visible=True)
	f3 = openpg.face([20,21,31,30])
	f4 = openpg.face([01,02,12,11,80,11])
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


        # Adding a pendent node
        #newnode = openpg.node(G, 99,99)
        #G.add_node(newnode)
        #G.add_edge(G.find_node_xy(1,1), newnode)
        #G.faces[1].add_edge(G.find_node_xy(1,1), newnode)
        #G[G.find_node_xy(1,1)][newnode]['faces'].add(G.faces[1])

        pend2 = openpg.node(G, 90,90)
        pend3 = openpg.node(G, 80,80)
        G.add_edge(pend2, G.find_node_xy(0,1))
        G.faces[0].add_edge(pend2, G.find_node_xy(0,1))
        G.add_edge(pend3, G.find_node_xy(0,1))
        G.faces[0].add_edge(pend3, G.find_node_xy(0,1))

        #print(G.faces[0])
        return G


