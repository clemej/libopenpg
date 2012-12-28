import pprint
import openpg
import isomorphic
import example_graphs as examples

p = pprint.PrettyPrinter(indent=4, depth=4)

G1 = examples.example_graph1()
G2 = examples.example_graph2()
G3 = examples.example_graph3()

#fo1 = G1.outer_face()
#fo1.set_initial_edge(G1.find_node_xy(0,0), G1.find_node_xy(1,0))
#fo2 = G2.outer_face()
#fo2.set_initial_edge(G2.find_node_xy(1,0), G2.find_node_xy(2,0))

G1.print_info(verbose=True)
G2.print_info(verbose=True)
G3.print_info(verbose=True)

print('---------- Normalizing -------------')
G1.normalize()
G1.print_info(verbose=True)

G2.normalize()
G2.print_info(verbose=True)

G3.normalize()
G3.print_info(verbose=True)

print('------ Checking isomorphism  -----')
print('For self:')
print('- Plane Isomorphic: ', isomorphic.check_plane_isomorphism(G1, G1))
print('- Sphere Isomoprhic:', isomorphic.check_sphere_isomorphism(G2, G2))
print('For others:')
print('- Plane Isomorphic: ', isomorphic.check_plane_isomorphism(G1, G2))
print('- Sphere Isomorphic:', isomorphic.check_sphere_isomorphism(G2, G1))

print('------ Checking for pattern matching -----')
print('For self:')
print('- pattern: ', isomorphic.check_pattern(G1, G1))
print('For others')
G4 = examples.example_graph4()
G4.normalize()
print('- pattern: ', isomorphic.check_pattern(G4,G1))
