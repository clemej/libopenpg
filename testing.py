import pprint
import openpg
import example_graphs as examples

G1 = examples.example_graph1()
G2 = examples.example_graph2()

G1.print_info(verbose=True)
G2.print_info(verbose=True)

print('---------- Normalizing -------------')
G1.normalize()
G1.print_info(verbose=True)

G2.normalize()
G2.print_info(verbose=True)

print('------ Checking isomorphism  -----')
print 'For self:'
print G1.check_plane_isomorphism(G1)
print G2.check_sphere_isomorphism(G2)
print 'for others:'
print G2.check_plane_isomorphism(G1)
print G1.check_sphere_isomorphism(G2)
