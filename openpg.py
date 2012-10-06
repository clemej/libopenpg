import networkx as nx
import cPickle as pickle

# Implement an open planar graph with faces, using the networkx graph library
# to handle the basic graph functions. 

def load(filename):
		fd = open(filename, 'rb')
		ret = pickle.load(fd)
		fd.close()
		return ret

def save(filename, graph):
		fd = open(filename, 'wb')
		pickle.dump(graph, fd)
		fd.close()


class node:
	def __init__(self, g, x, y):
		self.x = x
		self.y = y
		self.graph = g

	def equal(self, other):
		if other.graph == self.graph and \
				other.x == self.x and other.y == self.y:
			return True
		else:
			return False

	def __repr__(self):
		return '%s:(%d,%d)' % (self.graph, self.x, self.y)

	def __iter__(self):
		return (self.x, self.y)

class face:
	def __init__(self, graph, edgelist=[], adjacent=[], 
						visible=False, outer=False):
		self.graph = graph
		self.edgelist = []
		self.nodelist = set()
		if len(edgelist) > 2:
			self.edgelist = edgelist
			for n1,n2 in edgelist:
				self.add_edge(n1, n2)
		self.visible = visible
		self.outer = outer

	def __repr__(self):
		return 'Face(%s)(%s)' % (self.visible,self.nodelist)

	def add_edge(self, n1, n2):
		self.nodelist.add(n1)
		self.nodelist.add(n2)
		if not graph[n1][n2].get('faces', None):
			self.graph[n1][n2]['faces'] = set()
		self.graph[n1][n2]['faces'].add(self)

	def equal(self, face):
		if self.graph != face.graph:
			return False

		for n in self.nodelist:
			found = False
			for n2 in face.nodelist:
				if n.equal(n2):
					found = True
					break
			if not found:
				return False
		return True

	# find a common node
	# find the node in the graph
	# see if there's an edge to another common node
	# repeade for all common nodes
	def adjacent(self):
		ret = set()
		for n1,n2 in self.edgelist:
			for f in self.graph[n1][n2]['faces']:
				if f is not self:
					ret.add(f)

		return ret

	def is_adjacent(self, face):
		if self.graph != face.graph:
			return False

		return face in self.adjacent()

class openpg(nx.Graph):
	def __init__(self, data=None, name='', **attr):
		nx.Graph.__init__(self, data=data ,name=name,**attr)
		self.faces = []

	def add_face(self, face):
		self.faces.append(face)

	def find_node(self, n):
		for needle in self.nodes():
			if needle.equal(n):
				return needle
		return None

	def find_node_xy(self, x, y):
		for needle in self.nodes_iter():
			if needle.x == x and needle.y == y:
				return needle

	def find_edge(self, node1, node2):
		n1 = self.find_node(node1)
		n2 = self.find_node(node2)

		if n2 in self.neighbors(n1):
			return True
		return False

	def add_node_if_not_dup(self, node):
		if not self.find_node(node):
			self.add_node(node)

	def add_edge_if_not_dup(self, n1, n2):
		if not self.find_edge(n1, n2):
			self.add_edge(n1, n2)

	def add_edge(self, n1, n2, **kwargs):
		nx.Graph.add_edge(self, n1, n2, **kwargs)

	def outer_face(self):
		return filter(lambda x: x.outer, self.faces)[0]

	def pendents(self):
		return filter(lambda x: len(nx.neighbors(self, x)) == 1, 
							self.nodes_iter())

	def bridges(self):
		ret = []
		for edge in self.edges_iter():
			faces = []
			for face in self.faces:
				if edge in face.edgelist:
					faces.append(face)
			if len(faces) == 2 and not faces[0].visible and \
							not faces[1].visible:
				ret.append(edge)
		return ret

	def branches(self):
		ret = []
		for e in filter(lambda x: len(self[x[0]][x[1]]['faces']) == 1,
							self.edges_iter()):
			face = self[e[0]][e[1]]['faces'].pop()
			self[e[0]][e[1]]['faces'].add(face)
			if face.visible:
				continue
			if nx.degree(self,e[0]) > 1 and nx.degree(self,e[1]) > 1:
				ret.append(e)
		return ret

	def hinges(self):
		ret = []
		for node in filter(lambda x: nx.degree(self, x) > 3, 
							self.nodes_iter()):

			#adj_faces = []
			#for face in self.faces:
			#	if node in face.nodelist:
			#		adj_faces.append(face)
			test = []
			seen = []
			neighbors = nx.neighbors(self, node)
			node2 = neighbors[0]

			while len(seen) < len(neighbors):
				#print node,node2
				#print self[node][node2]
				face = self[node][node2]['faces'].pop()
				self[node][node2]['faces'].add(face)
				#while face == self.outer_face():
				#	face = self[node][node2]['faces'].pop()
				#	self[node][node2]['faces'].add(face)
				test.append(face.visible)
				seen.append(node2)

				node2list = filter(lambda x: x in face.nodelist and x not in seen, neighbors)
				if len(node2list) > 0:
					node2 = node2list[0]

			#print node,test
			newtest = [test[0]]
			for val in test:
				if val != newtest[-1]:
					newtest.append(val)
			if len(newtest) > 3:
				ret.append(node)

		return ret

	def print_info(self):
		print 'Nodes = %d' % (self.number_of_nodes())
		print 'Edges = %d' % (self.number_of_edges())
		print 'Faces = %d' % (len(self.faces))
		print 'Visible = (%d/%d)' % \
			(len(filter(lambda x: x.visible, self.faces)),
							len(self.faces))
		print 'Pendent nodes: %d' % len(self.pendents())
		print 'Bridges: %d' % len(self.bridges())
		print 'Branches: %d' % len(self.branches())
		print 'Hinges: %d' % len(self.hinges())


	def normalize(self):
		pass


if __name__ == '__main__':
	import sys

	G = load(sys.argv[1])
	G.print_info()
	print G.hinges()
	

