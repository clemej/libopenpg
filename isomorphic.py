import openpg
import pprint

def _get_outer_arcs(G):
        graph = G.graph
        return [arc for arc in graph.edges() if 
		graph[arc[0]][arc[1]].get('outer',None) ]

def _check_lemma4(f, G, OG, check_outer=True):
        graph = G.graph
        other = OG.graph
	#p = pprint.PrettyPrinter(indent=4, depth=4)
	#p.pprint(f)
	#if self._next(other, v) != self._next(graph, k) or \
        #print f
	for k in f.keys():
		#print k
		#print f[k]
                #print self._next(OG, f[k]), self._next(G, k)
		#if self._next(other, v) != self._next(graph, k) or \
                ognext = _next(OG, f[k])
                gnext = f[_next(G, k)]
                ogopp = _opp(f[k])
                gopp = f[_opp(k)]
                #print 'next: ', ognext, gnext 
                #print 'opp : ',ogopp, gopp

                #print 'next: ', ognext[0].equiv(gnext[0]) and ognext[1].equiv(gnext[1])
                #print 'opp: ', ogopp[0].equiv(gopp[0]) and ogopp[1].equiv(gopp[1])
                #print _next(OG, f[k]), _next(G, k)
		#if _next(OG, f[k]) != _next(G, k) or _opp(f[k]) != _opp(k):
                if ognext[0] is not gnext[0] or ognext[1] is not gnext[1]:
                        #print self._next(OG, f[k]), self._next(G, k)
        		#print('next is off')
			return False

                if ogopp[0] is not gopp[0] or ogopp[1] is not gopp[1]:
                        #print('opp is off')
                        return False

		if graph[k[0]][k[1]]['face'].visible != \
				other[f[k][0]][f[k][1]]['face'].visible:
                        #print graph[k[0]][k[1]]['arcface'], graph[k[0]][k[1]]['arcface'].visible
                        #print other[f[k][0]][f[k][1]]['arcface'], other[f[k][0]][f[k][1]]['arcface'].visible
                        #print 'visible is off'
			return False
		if check_outer and graph[k[0]][k[1]]['face'].outer \
        			!= other[f[k][0]][f[k][1]]['face'].outer:
			#print('faces are off')
			return False

	return True

def _trace_faces(G, initial):
	# build up a datastructure that keeps track of each face we've
	# seen, and the initial edge we used to find it, and whether
	# we've finished tracing it or not.
	# something like:
	# [ [face, [initial, done] , [face2, [initial, done]] ,...]
	# would like to use a python dict but it is diffcult to use
	# objects as keys. 
        graph = G.graph

        faces_info = []
        #print G.faces
	for f in G.faces:
		if f is G.outer_face():
			faces_info.append([f, [initial, False]])
		else:
			faces_info.append([f, [None, False]])

	while True:
		# find first entry that is not done and has an initial
		# edge
		for entry in faces_info:
			if entry[1][0] == None:
			        #print 'entry is none, skipping'
				continue
			if entry[1][1] == True:
				#print 'entry = True, skipping'
				continue

			face = entry[0]
			initedge = entry[1][0]

                        if face.outer == True:
                                edges = face.edges_in_order()
                        else:
			        edges = face.edges_in_order(
					initedge[0], initedge[1], initedge[0])

			#print edges
                        #print '======================================'
                        #print graph.nodes()
                        #print '--------------------------------------'
                        #print graph.edges()
                        #print '--------------------------------------'
                        #print edges
                        #print '--------------------------------------'
        		for e in edges:
                                #print e[0],e[1]
                                #print graph[e[0]][e[1]]
        			if not graph[e[0]][e[1]].get('arcface', None):
					#print 'working on: ', e
        				graph[e[0]][e[1]]['arcface'] = face
                                        if not face.outer and 'outer' in graph[e[0]][e[1]].keys():
                                                del graph[e[0]][e[1]]['outer']
					
                                        #print list(graph[e[0]][e[1]]['faces'])
					other_faces = filter(lambda x: x is not face, list(graph[e[0]][e[1]]['faces']))
					if len(other_faces) == 0:
						print 'edge ',e,'is pendent'
						continue

					other_face = other_faces[0]

					graph[e[1]][e[0]]['arcface'] = other_face
                                        if not other_face.outer and 'outer' in graph[e[1]][e[0]].keys() :
                                                del graph[e[1]][e[0]]['outer']

					for fi in faces_info:
					        #print fi
						if fi[0] is other_face and fi[1][0] == None:
							#print 'updating face initial edge'
							fi[1][0] = (e[1], e[0])

				else:
                                        pass
				
			entry[1][1] = True
                #print '=============================================================='
                #for fi in faces_info:
                #        print fi
        	#print faces_info
		ready_faces = filter(lambda x: x[1][1] == False, faces_info)
                #print ready_faces
        	#print len(ready_faces)
                if len(ready_faces) == 0:
                        return


def _fixup_edges(G):
	# 
	# We need to divide up the edges as to which face the
	# arc belongs to.  This is needed for the next function.
	# So, for each edge in the graph, look at the faces attribute
	# for each opposite arc.  If there is only one face between
	# them, then both arcs get the same face.  If there are two,
	# then we need to decide which arc gets which face.  
	# Faces have the property that there can only be one 
	# incoming arc and one outgoing arc per node associated with 
	# each face. 
	#

	# lets try this:
	#  Start with any outer face and an empty stack
	#  Set outer face on one of the two directed arcs, and
	#    the other face to the other arc.  Add other seen face
	#    to the queue.
	#  continue to trace outer edge with directed faces, adding
	#     any new faces to the queue
	#  when finished with outer, pop face from stack and do that
	#      face.  
	#  Finish when stack is empty.

        graph = G.graph
        for initial_edge in _get_outer_arcs(G):
                if graph[initial_edge[0]][initial_edge[1]].get('arcface',None):
                        # This graph has already been divided up, skip it.
                        return
		if len(graph[initial_edge[0]][initial_edge[1]]['faces']) > 1:
	                break

        initial_edge = G.outer_face().initial_edge

	_trace_faces(G, initial_edge)
	
        # XXXjc: I dont't think this is needed anymore...
	#for e in graph.edges_iter():
		# XXXjc: until I figure out why, some pendent edges don't
		# get labelled, so fix them here
	#	if graph[e[0]][e[1]].get('arcface', None) == None:
	#		for f in G.faces:
	#			if (e[0], e[1]) in f.edgelist or \
	#			    (e[1], e[0]) in f.edgelist:
        #                               print 'labelling missed edge'
	#				graph[e[0]][e[1]]['arcface'] = f


def _next(G, arc):
        graph = G.graph
        face = graph[arc[0]][arc[1]]['face']
	index = 0
	for edge in face.edges():
		if edge[0] == arc[0] and edge[1] == arc[1]:
			break
		index = index + 1

	return face.edges()[(index + 1) % len(face.edges())]


def _opp(arc):
        return ( arc[1], arc[0] )

def _traverse_and_build_matching(graph, other, arc, other_arc):
	"""
	Algorithm 29 from the paper
	"""
	f = {}
	f[arc] = other_arc
        #print 'arc = %s otherarc = %s', arc, other_arc
	stack = []
	stack.append(arc)
	while len(stack) > 0:
		a = stack.pop()
		if f.get(_next(graph, a), None) == None:
			f[_next(graph, a)] = _next(other,f[a])
			stack.append(_next(graph, a))
		if f.get(_opp(a), None) == None:
			f[_opp(a)] = _opp(f[a])
			stack.append(_opp(a))
	return f

def _traverse_visible_and_build_matching(graph, other, arc, other_arc):
	"""
	Algorithm 29 from the paper
	"""
	f = {}
	f[arc] = other_arc
        #print 'arc = %s otherarc = %s', arc, other_arc
	stack = []
	stack.append(arc)
	while len(stack) > 0:
		a = stack.pop()
		if f.get(_next(graph, a), None) == None:
			f[_next(graph, a)] = _next(other,f[a])
			stack.append(_next(graph, a))
                oa = _opp(a)
		if graph[oa[0]][oa[1]]['arcface'].visible and f.get(_opp(a), None) == None:
			f[_opp(a)] = _opp(f[a])
			stack.append(_opp(a))
	return f

def check_plane_isomorphism(G, OG):
	"""
	Algorithm 28 from the paper
	"""

        arc0 = G.outer_face().edges()[0]
        #print other_arcs
        #p = pprint.PrettyPrinter(indent=4, depth=4)
	for other_arc in OG.outer_face().edges():
		f = _traverse_and_build_matching(G, OG, arc0, other_arc)
                #p.pprint(f) 
		if _check_lemma4(f, G, OG):
			return True

	return False

def check_sphere_isomorphism(G, OG):
        arc0 = G.outer_face().edges()[0]

	for other_arc in OG.outer_face().edges():
		f = _traverse_and_build_matching(G, OG, arc0, other_arc)
		if _check_lemma4(f, G, OG, check_outer=False):
			return True

        return False

def check_pattern(P, G):
        pass
