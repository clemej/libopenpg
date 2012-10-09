import networkx as nx
import pprint

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
			for n1,n2 in edgelist:
				self.add_edge(n1, n2)
		self.visible = visible
		self.outer = outer

	def __repr__(self):
		return 'Face(%s)(%s)' % (self.visible,self.nodelist)

	def add_edge(self, n1, n2):
		self.edgelist.append((n1,n2))
		self.nodelist.add(n1)
		self.nodelist.add(n2)
		#if len(self.graph[n1][n2]['faces']) == 2:
			#print ('-------------')
			#print (self.graph[n1][n2]['faces'])
			#print (self)
			#raise(Exception('Adding third face? %s, %s' % (n1, n2)))
		self.graph[n1][n2]['faces'].add(self)
		if len(self.graph[n1][n2]['faces']) > 2:
			print ('-------------')
			print (self.graph[n1][n2]['faces'])
			print (self)
			raise(Exception('Adding third face? %s, %s' % (n1, n2)))

	def remove_node(self, node):
		self.nodelist.remove(node)
		newlist = []
		for e in self.edgelist:
			if e[0] == node or e[1] == node:
				continue
			newlist.append(e)
		self.edgelist = newlist

	def remove_edge(self, n1, n2):
		newedgelist = []
		for e in self.edgelist:
			if e[0] in [n1,n2] and e[1] in [n1,n2]:
				continue
			newedgelist.append(e)
		self.edgelist = newedgelist

	def merge(self, other, bridge):
		n1,n2 = bridge

		self.remove_edge(n1,n2)

		for e in other.edgelist:
			if e[0] in [n1,n2] and e[1] in [n1,n2]:
				continue

			newfaceset = set()
			for f in self.graph[e[0]][e[1]]['faces']:
				if f == other:
					continue
				newfaceset.add(f)
			self.graph[e[0]][e[1]]['faces'] = newfaceset
			self.add_edge(e[0], e[1])


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

	def remove_face(self, face):
		newfacelist = []
		for f in self.faces:
			if f == face:
				continue
			newfacelist.append(f)
		self.faces = newfacelist

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
		self[n1][n2]['faces'] = set()

	def outer_face(self):
		return filter(lambda x: x.outer, self.faces)[0]

	def pendents(self):
		return [x for x in self.nodes_iter() 
				if len(nx.neighbors(self, x)) == 1]

	def bridges(self):
		ret = []
		for edge in self.edges_iter():
			if len(self[edge[0]][edge[1]]['faces']) == 2:
				visible = False
				for f in self[edge[0]][edge[1]]['faces']:
					#print(edge,f.visible)
					if f.visible:
						visible = True
				if not visible:
					ret.append(edge)

		return ret

	def branches(self):
		ret = []
		for e in [x for x in self.edges_iter() 
				if len(self[x[0]][x[1]].get('faces',[])) == 1]:
			for f in self[e[0]][e[1]]['faces']:
				face = f
				break
			if face.visible:
				continue
			if nx.degree(self,e[0]) > 1 and \
					nx.degree(self,e[1]) > 1:
				ret.append(e)
		return ret

	def hinges(self):
		ret = []
		for node in [x for x in self.nodes_iter() 
						if nx.degree(self, x) > 3]:
			result = self._examine_hinge(node)
			if len(result) > 3:
				ret.append(node)
		return ret 

	def _examine_hinge(self, node):
		#print(node)

		# Will hold a list of tuples:
		# [visible?, [list of contig. faces sharing visible?],
		#   ending node]
		result = []

		ordered_neighbors = []
		neighbors = nx.neighbors(self, node)

		# Take first neighbor node returned to start
		curnode = neighbors[0]

		# needed bacuse you can't index into sets()
		for face in self[node][curnode]['faces']:
			curface = face
			break
			
		curres = [curface.visible, [curface], curnode]

		while len(ordered_neighbors) < len(neighbors):
			ordered_neighbors.append(curnode)

			# Add/skip an pendent nodes in this face
			pendent_neighbors_in_face = [x for x in
					curface.nodelist if x in neighbors and
					len(self[node][x]['faces']) == 1 and 
					x not in ordered_neighbors]
			ordered_neighbors += pendent_neighbors_in_face

			newnode = [x for x in curface.nodelist 
						if x in neighbors and 
						x not in ordered_neighbors]
			if len(newnode) == 0:
				# exit condition XXXjc: clean up
				continue
			else:
				newnode = newnode[0]

			#print(newnode,curnode)
			#print(self[node][newnode]['faces'])
			#print(self[node][curnode]['faces'])

			#print(self[node][newnode]['faces']-self[node][curnode]['faces'])

			for f in self[node][newnode]['faces'] - \
					self[node][curnode]['faces']:
				newface = f
				break


			if newface.visible == curres[0]:
				curres[1].append(newface)
			else:
				result.append(curres)
				curres = [newface.visible, [newface], newnode]

			curnode = newnode
			curface = newface

		result.append(curres)
			
		return result

	def eliminate_hinge(self, hinge, marker=1000000):
		hinfo = self._examine_hinge(hinge)

		# Make sure we're a hinge
		if len(hinfo) < 4:
			return

		#print(hinge)
		#print(hinfo)

		new_face = face(self, edgelist=[], visible=False)

		for area in hinfo:
			visible = area[0]
			# Create a new node
			newnode = node(self, 
				marker*(hinfo.index(area)+1)+hinge.x, 
				marker*(hinfo.index(area)+1)+hinge.y)

			# for each face in this segment, replace node with
			# newnode
			for f in area[1]:
				neighbors = [x for x in f.nodelist 
					if x in nx.neighbors(self, hinge)]
				for n in neighbors:
					#print('creating edge %s, %s' % \
					#	(newnode, n))
					self.add_edge(newnode, n)
					# copy existing attributes dict
					# XXXjc: This should be necessary to
					#        preserve other attributes;
					#        but it causes weird errors.
					#for k in self[hinge][n].keys():
					#	#print(k)
					#	if k != 'faces':
					#		continue
					#	self[newnode][n][k] = \
					#		self[hinge][n][k]

					#print(self[newnode][n]['faces'])
					if n.x > 1000:
						print('n = %s' % n)
					f.add_edge(newnode, n)
					new_face.add_edge(newnode, n)
					#print(self[newnode][n]['faces'])
					#pp = pprint.PrettyPrinter(indent=4, depth=4)
					#if newnode.x == 4000002:
					#pp.pprint(self[newnode][n])

				f.remove_node(hinge)

				#print(f)
				#print(f.edgelist)

		self.add_face(new_face)
		self.remove_node(hinge)
	
	def eliminate_bridges(self):
		bridges = self.bridges()
		while(len(bridges) > 0):
			n1,n2 = bridges[0]
			faces = list(self[n1][n2]['faces'])

			if faces[1].outer:
				kept_face = faces[1]
				other_face = faces[0]
			else:
				kept_face = faces[0]
				other_face = faces[1]

			kept_face.merge(other_face, (n1,n2))
			self.remove_face(other_face)
			self.remove_edge(n1,n2)

			bridges = self.bridges()
			#print(len(bridges))

	def eliminate_pendents(self):
		pendents = [x for x in self.pendents() if not
			list(self[x][nx.neighbors(self, x)]['faces'])[0].visible]
		print(len(pendents))

		while len(pendents) > 0:
			p = pendents[0]

			edge = (p, nx.neighbors(self, p))
			face = list(self[edge[0]][edge[1]]['faces'])[0]

			print(edge)
			print(face)

			face.remove_edge(edge[0], edge[1])
			self.remove_node(p)

			pendents = [x for x in self.pendents() if not
				list(self[x][nx.neighbors(self, x)]['faces'])[0].visible]
			print(len(pendents))


	def print_info(self):
		print('Nodes = %d' % (self.number_of_nodes()))
		print('Edges = %d' % (self.number_of_edges()))
		print('Faces = %d' % (len(self.faces)))
		print('Visible = (%d/%d)' % \
			(len([x for x in self.faces if x.visible]),
							len(self.faces)))
		print('Pendent nodes: %d' % len(self.pendents()))
		print('Bridges: %d' % len(self.bridges()))
		print('Branches: %d' % len(self.branches()))
		print('Hinges: %d' % len(self.hinges()))


	def normalize(self):
		self.remove_hinges()
		self.remove_bridges()
		self.remove_pendents()


if __name__ == '__main__':
	import sys

	G = load(sys.argv[1])
	G.print_info()
	print(G.hinges())
	

