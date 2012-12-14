import pprint
import openpg
import isomorphic
import example_graphs as examples

p = pprint.PrettyPrinter(indent=4, depth=4)

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
print '- Plane Isomorphic: ', isomorphic.check_plane_isomorphism(G1, G1)
print '- Sphere Isomoprhic:', isomorphic.check_sphere_isomorphism(G2, G2)
print 'other others:'
print '- Plane Isomorphic: ', isomorphic.check_plane_isomorphism(G1, G2)
print '- Sphere Isomorphic:', isomorphic.check_sphere_isomorphism(G2, G1)
