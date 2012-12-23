import sys
from collections import deque
import pprint

try:
	import networkx as nx
except:
	print("Please install the required python NetworkX library")
	sys.exit(1)

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

def rotleft(l, n):
	d = deque(l)
	d.rotate(-n)
	return list(d)

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
	def __init__(self, nodeids, name=None, G=None, 
						visible=False, outer=False):
		self.name = name
		self.G = G
		self.nodes = nodeids
		self.visible = visible
		self.outer = outer

	def __repr__(self):
		return 'Face(%s)(%s)' % (self.visible,self.nodes)

	def edges(self):
		ret = []
		for idx in range(len(self.nodes)-1):
			ret.append((self.nodes[idx], self.nodes[idx+1]))
		ret.append((self.nodes[-1],self.nodes[0]))
		return ret

class openpg():
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
		self.graph = nx.DiGraph(data = data, name = name, **attr)
		self.faces = []

	def add_face(self, face):
		""" Adds a face object to the graph """
		face.G = self

		for idx, n in enumerate(face.nodes[:-1]):
			self.graph.add_edge(
				face.nodes[idx], face.nodes[idx+1], face = face)

		self.graph.add_edge(face.nodes[-1], face.nodes[0], face = face)
		self.faces.append(face)

	def remove_face(self, face):
		""" Removes a face object from the graph """
		newfacelist = []
		for f in self.faces:
			if f is face:
				continue
			newfacelist.append(f)
		self.faces = newfacelist

	def outer_face(self):
		""" Return the outer face """
		return [x for x in self.faces if x.outer][0]

	def pendents(self):
		""" Return all pendent nodes (regardless of visibility) """
		return [x for x in self.graph.nodes_iter() 
				if len(nx.neighbors(self.graph, x)) == 1]

	def bridges(self):
		""" Return all bridge edges as list of (n1,n2) pairs """
		ret = []
		for edge in self.graph.edges_iter():
			n1 = edge[0]
			n2 = edge[1]
			f1 = self.graph[n1][n2]['face']
			f2 = self.graph[n2][n1]['face']

			if f1 is not f2 and not f1.visible and not f2.visible:
				# Needed to keep both edges in directed graph
				# from being appended.
				if (n2,n1) not in ret:
					ret.append(edge)
			
		return ret

	def branches(self):
		""" Return all branches as list of (n1,n2) pairs """
		ret = []
		for edge in self.graph.edges_iter():
			n1 = edge[0]
			n2 = edge[1]
			f1 = self.graph[n1][n2]['face']
			f2 = self.graph[n2][n1]['face']

			if f1 is f2:
				if f1.visible:
					continue

				if len(nx.neighbors(self.graph, n1)) > 1 and \
					len(nx.neighbors(self.graph, n2)) > 1:
					if (n2,n1) not in ret:
						ret.append(edge)
		return ret

	def hinges(self):
		""" Return a list of all detected hinge nodes """
		ret = []
		# Consider a node a /possible/ hinge if it meets two criteria:
		# - it has a degree of at least 3
		# - it is in two or more visible faces
		for node in [x for x in self.graph.nodes_iter() 
				if len(nx.neighbors(self.graph, x)) > 3]:

			neighbors = nx.neighbors(self.graph, node)
			faces = set()
			#print node
			for n in neighbors:
				f1 = self.graph[node][n]['face']
				f2 = self.graph[n][node]['face']
				if f1.visible:
					faces.add(f1)
				if f2.visible:
					faces.add(f2)

			if len(faces) < 2:
				continue
			
			#print 'sending to examine_hings'
			result,on = self._examine_hinge(node)
			if len(result) > 3:
				#pprint.pprint(on)
				#pprint.pprint(result)
				ret.append(node)
		return ret 

	def _examine_hinge(self, node):
		""" Examine a node to see if it is a hinge node. """
		#print(node)

		# Will hold a list of tuples:
		# [visible?, [list of contig. faces sharing visible?]]
		result = []

		ordered_neighbors = []
		neighbors = nx.neighbors(self.graph, node)
		#print 'neighbors: ', neighbors

		# Take first neighbor node returned to start
		curnode = neighbors[0]
	
		# grab the incoming edge in from that neighbor. 	
		curface = self.graph[curnode][node]['face']
		curres = [curface.visible, [curface]]

		while len(ordered_neighbors) < len(neighbors):
			ordered_neighbors.append(curnode)

			# Add/skip an pendent nodes in this face
			pendent_neighbors_in_face = [x for x in
					curface.nodes if x in neighbors and
					self.graph[node][x]['face'] == 
					self.graph[x][node]['face'] and 
					x not in ordered_neighbors]
			ordered_neighbors += pendent_neighbors_in_face

			newnode = [x for x in curface.nodes 
						if x in neighbors and 
						x not in ordered_neighbors]

			if len(newnode) == 0:
				# exit condition XXXjc: clean up
				continue
			else:
				newnode = newnode[0]

			newface = self.graph[newnode][node]['face']

			if newface.visible == curres[0]:
				curres[1].append(newface)
			else:
				result.append(curres)
				curres = [newface.visible, [newface]]

			curnode = newnode
			curface = newface

		result.append(curres)
		return result, ordered_neighbors

	def _eliminate_hinges(self):
		hinges = self.hinges()

		while len(hinges) > 0:
			h = hinges[0]
			self._eliminate_hinge(h)

			hinges = self.hinges()

	def _eliminate_hinge(self, hinge, marker=1000000):
		""" 
		Replace a hinge node with OPG-equivalent represetation, adding
		new nodes as necessary
		"""
		hinfo,on = self._examine_hinge(hinge)

		# Make sure we're a hinge
		if len(hinfo) < 4:
			return

		new_face = face([],visible=False)

		for area in hinfo:
			visible = area[0]
			# Create a new node
			newnode = marker + hinfo.index(area)+1
			self.graph.add_node(newnode)

			# for each face in this segment, replace node with
			# newnode
			for f in area[1]:
				neighbors = [x for x in f.nodes 
					if x in nx.neighbors(self.graph, hinge)]
				for n in neighbors:
					if self.graph[n][hinge]['face'] is f:
						self.graph.add_edge(n,newnode,face=f)
						self.graph.add_edge(newnode,n,
								face=new_face)
						
					else:
						self.graph.add_edge(newnode,n,face=f)
						self.graph.add_edge(n,newnode,
								face=new_face)

				f.nodes = [newnode if x == hinge else x \
							for x in f.nodes]

		
		# Construct the new face nodes list from the last newnode added
		start = newnode
		new_face.nodes = [start]
		# For each out_edge (n1,n2), look for the one with face of 
		# 'new_face' and return n2. 
		nextnode = [x for x in self.graph.out_edges(start) if
			self.graph[x[0]][x[1]]['face'] is new_face][0][1]
		while nextnode is not start:
			new_face.nodes.append(nextnode)
			nextnode = [x for x in self.graph.out_edges(nextnode)
				if self.graph[x[0]][x[1]]['face'] is 
								new_face][0][1]

		self.faces.append(new_face)
		self.graph.remove_node(hinge)

	def _eliminate_bridges(self):
		""" Remove all bridges """
		bridges = self.bridges()
		while(len(bridges) > 0):
			n1,n2 = bridges[0]
			faces = [self.graph[n1][n2]['face'],
				     self.graph[n2][n1]['face']]

			if faces[1].outer:
				kept_face = faces[1]
				other_face = faces[0]
				n2,n1 = bridges[0]
			else:
				kept_face = faces[0]
				other_face = faces[1]

			# Find n1,n2 in the face we're keeping
			lkf = len(kept_face.nodes)
			if n1 == kept_face.nodes[-1] and \
						n2 == kept_face.nodes[0]:
				kf_idx = lkf - 1
			else:
				for kf_idx in range(lkf):
					if kept_face.nodes[kf_idx:kf_idx+2] \
								== [n1, n2]:
						break

			lof = len(other_face.nodes)
			if n2 == other_face.nodes[-1] and \
						n1 == other_face.nodes[0]:
				of_idx = lof - 1
			else:
				for of_idx in range(lof):
					if other_face.nodes[of_idx:of_idx+2] \
								== [n2, n1]:
						break
			of_idx = of_idx + 1

			# Generate the new face node's list 
			kept_face.nodes = kept_face.nodes[:kf_idx+1] + \
				rotleft(other_face.nodes, of_idx)[1:-1] + \
				kept_face.nodes[kf_idx+1:]

			# Remove both edges
			self.graph.remove_edge(n1, n2)
			self.graph.remove_edge(n2, n1)

			# Relabel all edges with the new face
			for idx, n in enumerate(kept_face.nodes[:-1]):
				self.graph[kept_face.nodes[idx]]\
					  [kept_face.nodes[idx+1]]\
					  ['face'] = kept_face

			self.graph[kept_face.nodes[-1]][kept_face.nodes[0]]\
							['face'] = kept_face

			# Remove the old face from the list of faces
			self.faces.remove(other_face)

			bridges = self.bridges()

	def _eliminate_pendents(self):
		""" Remove all pendent nodes in non-visible faces """
		#find each pendent in an invisible face
		pendents = [x for x in self.pendents() if not
				self.graph[x][nx.neighbors(self.graph, x)[0]]\
							['face'].visible]

		while len(pendents) > 0:
			p = pendents[0]
			other = nx.neighbors(self.graph, p)[0]
			face = self.graph[p][other]['face']
			
			self.graph.remove_edge(p,other)
			self.graph.remove_edge(other,p)
			self.graph.remove_node(p)

			for idx in range(len(face.nodes)):
				if face.nodes[idx:idx+2] == [p, other]:
					break

			face.nodes = face.nodes[0:idx] + face.nodes[idx+2:]

			pendents = [x for x in self.pendents() if not
				self.graph[x][nx.neighbors(self.graph, x)[0]]\
							['face'].visible]

	def print_info(self, verbose=False):
		pp = pprint.PrettyPrinter(indent=2, depth=4)
		print('Nodes = %d' % (self.graph.number_of_nodes()))
		print('Edges = %d' % (self.graph.number_of_edges()))
		print('Faces = %d' % (len(self.faces)))
		#if verbose:
		#	pp.pprint(self.faces)
		print('Visible = (%d/%d)' % \
			(len([x for x in self.faces if x.visible]),
							len(self.faces)))
		print('Pendent nodes: %d' % len(self.pendents()))
		if verbose:
			pp.pprint(self.pendents())
		print('Bridges: %d' % len(self.bridges()))
		if verbose:
			pp.pprint(self.bridges())
		print('Branches: %d' % len(self.branches()))
		if verbose:
			pp.pprint(self.branches())
		print('Hinges: %d' % len(self.hinges()))
		if verbose:
			pp.pprint(self.hinges())


	def normalize(self):
		""" 
		Normalize the graph as described in the paper:

		- Remove all hinges
		- Remove all bridges
		- Remove all pendents edges in non-visible faces
		"""
		self._eliminate_hinges()
		self._eliminate_bridges()
		self._eliminate_pendents()

if __name__ == '__main__':
	import sys

	G = load(sys.argv[1])
	G.print_info()
	print(G.hinges())
	

