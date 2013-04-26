import sys
import pickle
from collections import deque
import pprint
import traceback
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

def load_facefmt(filename, convfunc):
	fd = open(filename, 'rb')

	G = openpg()
	for line in fd.readlines():
		if line[0] == '#':
			continue
		if line[0].upper() == 'N':
			G.name = line.split()[1]
		if line[0].upper() == 'F':
			visible = False
			outer = False
			labels = {}
			finfo = line.split()
			if 'outer' in finfo:
				outer = True
				finfo.remove('outer')
			if 'visible' in finfo:
				visible = True
				finfo.remove('visible')
			for e in finfo:
				if not e.startswith('labels='):
					continue
				kvs = e.split('"')[1]
				for ent in kvs.split(':'):
					if not '=' in ent:
						continue
					k,v = ent.split('=')
					labels[k] = v
				finfo.remove(e)
			nodes = map(lambda x: convfunc(x), finfo[1:])

			f = face(nodes, labels=labels, visible=visible, outer=outer)
			G.add_face(f)
			
	fd.close()
	return G

def save_facefmt(graph, filename, convfunc):
	fd = open(filename, 'wb')
	fd.write('VERSION 2\n')
	if graph.name != '':
		fd.write('N %s\n' % graph.name)
	for face in graph.faces:
		fd.write('F ')
		for node in face.nodes:
			fd.write('%s ' % convfunc(node))
		fd.write('labels="')
		if len(face.labels.keys()) > 0:
			#print face.labels
			for k,v in face.labels.iteritems():
				fd.write(str(k) + '=' + str(v) + ':')
		fd.write('" ')	
		if face.visible:
			fd.write('visible ')
		if face.outer:
			fd.write('outer ')
		fd.write('\n')
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
	def __init__(self, nodeids, name=None, G=None, labels={},
						visible=False, outer=False):
		self.name = name
		self.G = G
		self.nodes = nodeids
		self.visible = visible
		self.outer = outer
		self.index = 0
		self.labels = labels

	def __repr__(self):
		return 'Face(%s)(%s)' % (self.visible,self.nodes)

	def edges(self):
		ret = []
		if len(self.nodes) < 2:
			return ret
		for idx in range(len(self.nodes)-1):
			ret.append((self.nodes[idx], self.nodes[idx+1]))
		ret.append((self.nodes[-1],self.nodes[0]))
		return ret

	def copy(self):
		return face(self.nodes, name=self.name, labels=self.labels, 
				visible=self.visible, outer=self.outer)

	def equiv(self, other):
		if len(other.nodes) != len(self.nodes):
			return False
	
		samenodes = False
		for i in range(len(self.nodes)):
			if self.nodes == rotleft(other.nodes, i):
				samenodes = True
		
		if samenodes and other.outer == self.outer and \
						other.visible == self.visible:
			return True

		return False

	def adjacent(self):
		ret = set()
		for e in self.edges():
			ret.add(self.G.graph[e[1]][e[0]]['face'])

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
		self.name = name

	def add_face(self, face):
		""" Adds a face object to the graph """
		face.G = self

		if len(face.nodes) > 0:
			for idx, n in enumerate(face.nodes[:-1]):
				self.graph.add_edge(
					face.nodes[idx], face.nodes[idx+1], 
					face = face)

			self.graph.add_edge(face.nodes[-1], face.nodes[0], 
					face = face)
			#print '--- added ', face.nodes[-1], face.nodes[0]

		face.index = len(self.faces)
		self.faces.append(face)

	def dual(self, labelfunc=None):
		dg = nx.Graph(name='%s-dual' % self.graph.name)
		for face in self.faces:
			for fadj in face.adjacent():
				#if self.outer_face() not in [face, fadj]:
				dg.add_edge(face, fadj)
				if labelfunc:
					x,y = labelfunc(face, fadj)
					dg[face][fadj][x] = y

		return dg

	def remove_face(self, face):
		""" Removes a face object from the graph """
		return self.faces.remove(face)

	def outer_face(self):
		""" Return the outer face """
		return [x for x in self.faces if x.outer][0]

	def _adjacent_visible(self, face):
		return {x for x in list(face.adjacent()) if x.visible}

	def contiguous_visible_faces(self, face, ret=set()):
		if not face.visible or face in ret:
			return ret

		ret.add(face)

		newfaces = self._adjacent_visible(face)
		for f in list(newfaces):
			ret.update(self.contiguous_visible_faces(f, ret=ret))
		
		ret.update(newfaces)
		return ret

	def pendents(self):
		""" Return all pendent nodes (regardless of visibility) """
		return [x for x in self.graph.nodes_iter() 
				if len(nx.neighbors(self.graph, x)) == 1]

	def bridges(self):
		""" Return all bridge edges as list of (n1,n2) pairs """
		ret = []
		for edge in self.graph.edges_iter():
			try:
				n1 = edge[0]
				n2 = edge[1]
				f1 = self.graph[n1][n2]['face']
				f2 = self.graph[n2][n1]['face']
			except:
				print("error in stuff and things")
				print n1,n2,f1
				print self.outer_face()
				print self.graph.out_degree([n1])
				print self.graph.out_degree([n2])
				print self.graph.in_degree([n1])
				print self.graph.in_degree([n2])
				print traceback.format_exc()
				raise Exception("error")

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


