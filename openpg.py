import sys
import pprint

try:
	import networkx as nx
except:
	print("Please install the required python NetworkX library")
	sys.exit(0)

# Implement an open planar graph with faces, using the networkx graph library
# to handle the basic graph functions. 

def load(filename):
	"""
	Loads a previously saved graph from filename.

	The graph is a pickled python object for now until there's a 
	better preresentation.
	"""
	fd = open(filename, 'rb')
	ret = pickle.load(fd)
	fd.close()
	return ret

def save(graph, filename):
	"""
	Saves a graph object out to a file using python pickling. 

	The graph is a pickled python object for now until there's a 
	better preresentation.
	"""
	fd = open(filename, 'wb')
	pickle.dump(graph, fd)
	fd.close()


class node:
	"""
	Class for implementing a node in the open planar graph.

	Any node class can be used, but is must provide an equal() method
	that can be used to compare two nodes.  

	This impletation of a node has an x and y attribute that must be 
	uniqie, because this is intended for image processing.  

	XXXc: Make this more generic.
	"""
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
	"""
	A face in a graph.  A face is defined by a set of edges and nodes
	that surround a physically contiguous space.  Each edge keeps track
	of the faces it is adjacent to as a networkx edge attribute.

	Faces can be either 'visible' or 'invisible', indicated by
	the 'visible' attribute. 

	One space in the graph is the "outer" face, which represents all the
	space beyond the boundary of the graph.  Indicated with the 'outer'
	attribute.
	"""
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
		""" Add an edge to an existing face """
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
		""" Remove a node and all connected face edges """
		self.nodelist.remove(node)
		newlist = []
		for e in self.edgelist:
			if e[0] == node or e[1] == node:
				continue
			newlist.append(e)
		self.edgelist = newlist

	def remove_edge(self, n1, n2):
		""" Remove an edge, this leaves the nodes """
		newedgelist = []
		for e in self.edgelist:
			if e[0] in [n1,n2] and e[1] in [n1,n2]:
				#print('ignoring edge %s' % str(e))
				continue
			newedgelist.append(e)
		#print(self.edgelist)
		#print(newedgelist)
		self.edgelist = newedgelist

	def merge(self, other, bridge):
		""" 
		Merge an adjacent face with this face.  bridge is a 
		(node1,node2) edge definition of the shared edge. 

		- The bridge edge is removed from both faces. 
		- All nodes in the 'other' face are added to this face.
		- All edges in the 'other' face are added to this face.
		- All edge attrbiutes in new face that point to 'other' are 
		  updated to point to this face.
		"""
		n1,n2 = bridge

		self.remove_edge(n1,n2)

		for e in other.edgelist:
			#print (e)
			other.remove_edge(n1,n2)
			if e[0] in [n1,n2] and e[1] in [n1,n2]:
				#print('Ingoring other bridge edge %s' % str(e))
				continue

			newfaceset = set()
			for f in list(self.graph[e[0]][e[1]]['faces']):
				if f == other:
					print('removing face %s' % f)
					continue
				newfaceset.add(f)
			self.graph[e[0]][e[1]]['faces'] = newfaceset
			#print(self.graph[e[0]][e[1]]['faces'])
			self.add_edge(e[0], e[1])


	def equal(self, face):
		"""
		Compare two faces to see if they are equivalent

		XXXjc: needed anymore? probablt broken
		"""
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
		"""
		Find all adjacent faces to a particular face. 

		XXXjc: Maybe use edge attributes? probably broken now.
		"""
		ret = set()
		for n1,n2 in self.edgelist:
			for f in self.graph[n1][n2]['faces']:
				if f is not self:
					ret.add(f)

		return ret

	def is_adjacent(self, face):
		""" determine if another face is adjacent to this face. """
		if self.graph != face.graph:
			return False

		return face in self.adjacent()



class openpg(nx.Graph):
	""" 
	A graph structure that implements an Open Planar Graph as
	described in:

	Polynomial Algorithms for Open Plane Graph and Subgraph Isomorphism 
	by de la Higuera, Janodet, Samuel, et al.

	This is implemented using the NetworkX library for basic graph 
	functions of edges and nodes, so this is a subclass of nx.Graph().
	This has the nice benefit of using NX's conversion routines to convert
	between directed and undirecte graphs. 

	Face objects are added to this class as a list, and pointers to faces
	are maintained as NX edge attribute "faces", therefore you must
	not add your own edge attribute "faces" to the graph.

	Functions exist to "normalize" the graph and search for isomorphism. 
	"""
	def __init__(self, data=None, name='', **attr):
		nx.Graph.__init__(self, data=data ,name=name,**attr)
		self.faces = []

	def add_face(self, face):
		""" Adds a face object to the graph """
		self.faces.append(face)

	def remove_face(self, face):
		""" Removes a face object from the graph """
		newfacelist = []
		for f in self.faces:
			if f == face:
				continue
			newfacelist.append(f)
		self.faces = newfacelist

	def find_node(self, n):
		""" Find and return a particular node """
		for needle in self.nodes_iter():
			if needle.equal(n):
				return needle
		return None

	def find_node_xy(self, x, y):
		""" 
		Convenience function to find nodes by x,y attribute 

		This is specific to the above node implementation.
		"""
		for needle in self.nodes_iter():
			if needle.x == x and needle.y == y:
				return needle

	def find_edge(self, node1, node2):
		""" Find if an edge exists within the graph """
		n1 = self.find_node(node1)
		n2 = self.find_node(node2)

		if n2 in self.neighbors(n1):
			return True
		return False

	def add_node_if_not_dup(self, node):
		""" Add node only if its not a duplicate """
		if not self.find_node(node):
			self.add_node(node)

	def add_edge_if_not_dup(self, n1, n2):
		""" Add edge only it doesn't already exist """
		if not self.find_edge(n1, n2):
			self.add_edge(n1, n2)

	def add_edge(self, n1, n2, **kwargs):
		""" Add an edge to the graph, initialize faces attribute """
		nx.Graph.add_edge(self, n1, n2, **kwargs)
		self[n1][n2]['faces'] = set()

	def outer_face(self):
		""" Return the outer face """
		return filter(lambda x: x.outer, self.faces)[0]

	def pendents(self):
		""" Return all pendent nodes (regardless of visibility) """
		return [x for x in self.nodes_iter() 
				if len(nx.neighbors(self, x)) == 1]

	def bridges(self):
		""" Return all bridge edges as list of (n1,n2) pairs """
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
		""" Return all branches as list of (n1,n2) pairs """
		ret = []
		for e in [x for x in self.edges_iter() 
				if len(self[x[0]][x[1]].get('faces',[])) == 1]:
			face = list(self[e[0]][e[1]]['faces'])[0]
			if face.visible:
				continue
			if nx.degree(self,e[0]) > 1 and \
					nx.degree(self,e[1]) > 1:
				ret.append(e)
		return ret

	def hinges(self):
		""" Return a list of all detected hinge nodes """
		ret = []
		for node in [x for x in self.nodes_iter() 
						if nx.degree(self, x) > 3]:
			result = self._examine_hinge(node)
			if len(result) > 3:
				ret.append(node)
		return ret 

	def _examine_hinge(self, node):
		""" Examine a node to see if it is a hinge node. """
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
		curface = list(self[node][curnode]['faces'])[0]
		#for face in self[node][curnode]['faces']:
		#	curface = face
		#	break
			
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

			print(node,newnode,curnode)
			print(self[node][newnode]['faces'])
			print(self[node][curnode]['faces'])

			print(self[node][newnode]['faces'] - 
					self[node][curnode]['faces'])

			newface = list(self[node][newnode]['faces'] -
					self[node][curnode]['faces'])[0]

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
		""" 
		Replace a hinge node with OPG-equivalent represetation, adding
		new nodes as necessary
		"""
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
					#pp = pprint.PrettyPrinter(indent=4, 
					#			depth=4)
					#if newnode.x == 4000002:
					#pp.pprint(self[newnode][n])

				f.remove_node(hinge)

				#print(f)
				#print(f.edgelist)

		self.add_face(new_face)
		self.remove_node(hinge)
	
	def eliminate_bridges(self):
		""" Remove all bridges """
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
		""" Remove all pendent nodes in non-visible faces """
		#find each pendent in an invisible face
		pendents = [x for x in self.pendents() if not
			list(self[x][nx.neighbors(self, x)[0]]
						['faces'])[0].visible]
		print(len(pendents))

		while len(pendents) > 0:
			p = pendents[0]

			edge = (p, nx.neighbors(self, p)[0])
			face = list(self[edge[0]][edge[1]]['faces'])[0]

			#print(edge)
			#print(face)

			face.remove_edge(edge[0], edge[1])
			self.remove_node(p)

			pendents = [x for x in self.pendents() if not
				list(self[x][nx.neighbors(self, x)[0]]['faces'])[0].visible]
			print(pendents)
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
		""" 
		Normalize the graph as described in the paper:

		- Remove all hinges
		- Remove all bridges
		- Remove all pendents edges in non-visible faces
		"""
		self.remove_hinges()
		self.remove_bridges()
		self.remove_pendents()


if __name__ == '__main__':
	import sys

	G = load(sys.argv[1])
	G.print_info()
	print(G.hinges())
	

