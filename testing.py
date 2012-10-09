import pprint
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

print(G.nodes())

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
G.add_edge(G.find_node_xy(1,0), G.find_node_xy(1,1))
G.add_edge(G.find_node_xy(2,0), G.find_node_xy(2,1))
G.add_edge(G.find_node_xy(3,0), G.find_node_xy(3,1), data={'outer': True})
G.add_edge(G.find_node_xy(0,1), G.find_node_xy(0,2), data={'outer': True})
G.add_edge(G.find_node_xy(1,1), G.find_node_xy(1,2))
G.add_edge(G.find_node_xy(2,1), G.find_node_xy(2,2))
G.add_edge(G.find_node_xy(3,1), G.find_node_xy(3,2), data={'outer': True})

print(G.edges())

outer_edges = [x for x in G.edges_iter() if G[x[0]][x[1]].get('data',{'outer': False}).get('outer',False)]
#print(G[G.find_node_xy(0,0)][G.find_node_xy(0,1)])
#print(outer_edges)
outer_face = openpg.face(G, edgelist=outer_edges, visible=False, outer=True)
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
print(G.bridges())

# Adding a pendent node
newnode = openpg.node(G, 99,99)
G.add_node(newnode)
G.add_edge(G.find_node_xy(1,1), newnode)
G.faces[1].add_edge(G.find_node_xy(1,1), newnode)
#G[G.find_node_xy(1,1)][newnode]['faces'].add(G.faces[1])

pend2 = openpg.node(G, 90,90)
pend3 = openpg.node(G, 80,80)
G.add_edge(pend2, G.find_node_xy(0,1))
G.faces[0].add_edge(pend2, G.find_node_xy(0,1))
G.add_edge(pend3, G.find_node_xy(0,1))
G.faces[0].add_edge(pend3, G.find_node_xy(0,1))

#print(G.faces[0])

G.print_info()
pp = pprint.PrettyPrinter(indent=8)
#pp.pprint(G.hinges())


G.eliminate_hinge(G.hinges()[0])

for edge in G.edges_iter():
	if len(G[edge[0]][edge[1]]['faces']) == 0:
		print(edge,G[edge[0]][edge[1]]['faces'])

print('--------------')

G.print_info()

G.eliminate_bridges()
