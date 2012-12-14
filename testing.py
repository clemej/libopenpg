import pprint
import openpg
import isomorphic
import example_graphs as examples

p = pprint.PrettyPrinter(indent=4, depth=4)

G1 = examples.example_graph1()
p.pprint(G1.graph.nodes())
G2 = examples.example_graph2()
p.pprint(G2.graph.nodes())

G1.print_info(verbose=True)
G2.print_info(verbose=True)

print('---------- Normalizing -------------')
G1.normalize()
G1.print_info(verbose=True)

G2.normalize()
G2.print_info(verbose=True)

print('------ Checking isomorphism  -----')
print 'For self:'
print isomorphic.check_plane_isomorphism(G1, G1)
print isomorphic.check_sphere_isomorphism(G2, G2)
print 'for others:'
print isomorphic.check_plane_isomorphism(G1, G2)
print isomorphic.check_sphere_isomorphism(G2, G1)
