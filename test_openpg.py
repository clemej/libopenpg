import unittest
import openpg
import example_graphs
import isomorphic

class test_openpg(unittest.TestCase):
	"""
	Unittests for openpg library
	"""
	def setUp(self):
		self.G1 = example_graphs.example_graph1()
		self.G2 = example_graphs.example_graph2()
		self.G3 = example_graphs.example_graph3()

	def test_save_load(self):
		openpg.save(self.G1, '/tmp/openpg_test.save')
		g1 = openpg.load('/tmp/openpg_test.save')
		self.assertEqual(self.G1.name, g1.name)
		self.assertEqual(len(self.G1.faces), len(g1.faces))

	def test_find_pendents(self):
		g1 = self.G1.pendents()
		self.assertEqual(len(g1),1)

	def test_find_bridges(self):
		g1 = self.G1.bridges()
		self.assertEqual(len(g1),4)
	
	def test_find_branches(self):
		g2 = self.G2.branches()
		self.assertEqual(len(g2),1)

	def test_find_hinges(self):
		g3 = self.G3.hinges()
		self.assertEqual(len(g3),1)

	def test_normalize(self):
		for g in [self.G1, self.G2, self.G3]:
			g.normalize()
			self.assertEqual(g.hinges(), [])
			self.assertEqual(g.branches(), [])
			self.assertEqual(g.bridges(), [])

	def test_outer_face(self):
		g = openpg.openpg()
		f1 = openpg.face([], outer=True)
		f2 = openpg.face([], visible=True)
		g.add_face(f1)
		g.add_face(f2)
		self.assertIs(g.outer_face(), f1)

	def test_remove_face(self):
		g = openpg.openpg()
		f1 = openpg.face([], outer=True)
		f2 = openpg.face([], visible=True)
		g.add_face(f1)
		g.add_face(f2)
		self.assertEqual(len(g.faces), 2)
		g.remove_face(f1)
		self.assertEqual(len(g.faces), 1)

	def test_print_face(self):
		f1 = openpg.face([], outer=True)
		self.assertEqual(str(f1), 'Face(False)([])')

	def test_face_edges_lessthan_2(self):
		f1 = openpg.face([])
		self.assertEqual(f1.edges(), [])

	def test_printout(self):
		self.G1.print_info(verbose=True)

	def test_plane_iso_same(self):
		self.G1.normalize()
		self.assertTrue(
			isomorphic.check_plane_isomorphism(self.G1, self.G1))

	def test_plane_iso_diff(self):
		self.G1.normalize()
		self.G2.normalize()
		self.assertTrue(
			isomorphic.check_plane_isomorphism(self.G2, self.G1))

	def test_sphere_iso_same(self):
		self.G1.normalize()
		self.assertTrue(
			isomorphic.check_sphere_isomorphism(self.G1, self.G1))

	def test_sphere_iso_diff(self):
		self.G1.normalize()
		self.G2.normalize()
		self.assertTrue(
			isomorphic.check_sphere_isomorphism(self.G2, self.G1))

	def test_plane_iso_false(self):
		self.G1.normalize()
		self.G3.normalize()
		self.assertFalse(
			isomorphic.check_plane_isomorphism(self.G3, self.G1))

	def test_sphere_iso_false(self):
		self.G1.normalize()
		self.G3.normalize()
		self.assertFalse(
			isomorphic.check_sphere_isomorphism(self.G3, self.G1))

	def test_check_pattern_same(self):
		self.G1.normalize()
		self.assertTrue(isomorphic.check_pattern(self.G1, self.G1))

	

	

if __name__ == '__main__':
	unittest.main()

