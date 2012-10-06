import openpg

#
# o---o---o---o
# | 1 | 2 | 3 |   Nodes numbered on grid from 0,0 to 3,2. left to right,  
# o---o---o---o                                           top to bottom.
# | 4 | 5 | 6 |   f1, f2, f6 are visible. 
# o---o---o---o
#

G = openpg.openpg(name='square6')

for x in range(0,4):
	for y in range(0,3):
		node = openpg.node(G, x, y)
		G.add_node(node)

print G.nodes()

G.add_edge(G.find_node_xy(0,0), G.find_node_xy(1,0), data={'outer': True})
G.add_edge(G.find_node_xy(1,0), G.find_node_xy(2,0), data={'outer': True})
G.add_edge(G.find_node_xy(2,0), G.find_node_xy(3,0), data={'outer': True})
G.add_edge(G.find_node_xy(0,1), G.find_node_xy(1,1))
G.add_edge(G.find_node_xy(1,1), G.find_node_xy(2,1))
G.add_edge(G.find_node_xy(2,1), G.find_node_xy(3,1))
G.add_edge(G.find_node_xy(0,2), G.find_node_xy(1,2), data={'outer': True})
G.add_edge(G.find_node_xy(1,2), G.find_node_xy(2,2), data={'outer': True})
G.add_edge(G.find_node_xy(2,2), G.find_node_xy(3,2), data={'outer': True})

G.add_edge(G.find_node_xy(0,0), G.find_node_xy(0,1), data={'outer': True})
G.add_edge(G.find_node_xy(1,0), G.find_node_xy(1,1), data={'outer': True})
G.add_edge(G.find_node_xy(2,0), G.find_node_xy(2,1))
G.add_edge(G.find_node_xy(3,0), G.find_node_xy(3,1))
G.add_edge(G.find_node_xy(0,1), G.find_node_xy(0,2), data={'outer': True})
G.add_edge(G.find_node_xy(1,1), G.find_node_xy(1,2))
G.add_edge(G.find_node_xy(2,1), G.find_node_xy(2,2))
G.add_edge(G.find_node_xy(3,1), G.find_node_xy(3,2), data={'outer': True})

print G.edges()

outer_edges = filter(lambda x: G[x[0]][x[1]].get('outer',False),G.edges_iter())
outer_face = openpg.face(G, outer_edges, visible=False, outer=True)
G.add_face(outer_face)

# f1
G.add_face(openpg.face(G, edgelist=[
	(G.find_node_xy(0,0), G.find_node_xy(0,1)),
	(G.find_node_xy(1,0), G.find_node_xy(1,1)),
	(G.find_node_xy(0,0), G.find_node_xy(1,0)),
	(G.find_node_xy(0,1), G.find_node_xy(1,1))
	], visible=True))

# f2
G.add_face(openpg.face(G, edgelist=[
	(G.find_node_xy(1,0), G.find_node_xy(1,1)),
	(G.find_node_xy(2,0), G.find_node_xy(2,1)),
	(G.find_node_xy(1,0), G.find_node_xy(2,0)),
	(G.find_node_xy(1,1), G.find_node_xy(2,1))
	], visible=True))

# f3
G.add_face(openpg.face(G, edgelist=[
	(G.find_node_xy(3,0), G.find_node_xy(3,1)),
	(G.find_node_xy(2,0), G.find_node_xy(2,1)),
	(G.find_node_xy(2,0), G.find_node_xy(3,0)),
	(G.find_node_xy(2,1), G.find_node_xy(3,1))
	], visible=False))

# f4
G.add_face(openpg.face(G, edgelist=[
	(G.find_node_xy(0,1), G.find_node_xy(0,2)),
	(G.find_node_xy(1,1), G.find_node_xy(1,2)),
	(G.find_node_xy(0,1), G.find_node_xy(1,1)),
	(G.find_node_xy(0,2), G.find_node_xy(1,2))
	], visible=False))

# f5
G.add_face(openpg.face(G, edgelist=[
	(G.find_node_xy(1,1), G.find_node_xy(1,2)),
	(G.find_node_xy(2,1), G.find_node_xy(2,2)),
	(G.find_node_xy(1,1), G.find_node_xy(2,1)),
	(G.find_node_xy(1,2), G.find_node_xy(2,2))
	], visible=False))

# f6
G.add_face(openpg.face(G, edgelist=[
	(G.find_node_xy(3,1), G.find_node_xy(3,2)),
	(G.find_node_xy(2,1), G.find_node_xy(2,2)),
	(G.find_node_xy(2,1), G.find_node_xy(3,1)),
	(G.find_node_xy(2,2), G.find_node_xy(3,2))
	], visible=True))

G.print_info()
print G.branches()

# Adding a pendent node
newnode = G.add_node(G, x=99, y=99)
G.add_edge(G.find_node_xy(0,0), newnode)

