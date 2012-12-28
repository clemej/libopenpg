import openpg
import pprint

def _check_lemma4(f, G, OG, check_outer=True):
	graph = G.graph
	other = OG.graph
	for k in list(f.keys()):
		ognext = _next(OG, f[k])
		gnext = f[_next(G, k)]
		ogopp = _opp(f[k])
		gopp = f[_opp(k)]
		if ognext[0] is not gnext[0] or ognext[1] is not gnext[1]:
			return False

		if ogopp[0] is not gopp[0] or ogopp[1] is not gopp[1]:
                        return False

		if graph[k[0]][k[1]]['face'].visible != \
				other[f[k][0]][f[k][1]]['face'].visible:
			return False
		if check_outer and graph[k[0]][k[1]]['face'].outer \
				!= other[f[k][0]][f[k][1]]['face'].outer:
			return False

	return True

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
	Algorithm 31 from the paper
	"""
	f = {}
	f[arc] = other_arc
	stack = []
	stack.append(arc)
	while len(stack) > 0:
		a = stack.pop()
		if f.get(_next(graph, a), None) == None:
			f[_next(graph, a)] = _next(other,f[a])
			stack.append(_next(graph, a))
		oa = _opp(a)
		if graph.graph[oa[0]][oa[1]]['face'].visible and \
						f.get(_opp(a), None) == None:
			f[_opp(a)] = _opp(f[a])
			stack.append(_opp(a))
	return f

def check_plane_isomorphism(G, OG):
	"""
	Algorithm 28 from the paper
	"""

	arc0 = G.outer_face().edges()[0]
	for other_arc in OG.outer_face().edges():
		f = _traverse_and_build_matching(G, OG, arc0, other_arc)
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
	parcs = [x for x in P.graph.edges_iter() if 
			P.graph[x[0]][x[1]]['face'].visible]
	arc0 = parcs[0]

	other_arcs = [x for x in G.graph.edges_iter() if
			G.graph[x[0]][x[1]]['face'].visible]

	for arc in other_arcs:
		f = _traverse_visible_and_build_matching(P,G,arc0,arc)
		W = set()
		for a in parcs:
			fa = f[a]
			W.add(G.graph[fa[0]][fa[1]]['face'])

		visible_faces = {x for x in G.faces if x.visible}
		if W <= visible_faces: # and W is contiguous
			Gprime = openpg.openpg()
			for face in G.faces:
				Gprime.add_face(face.copy())
			for face in Gprime.faces:
				face.visible = False
				for w in list(W):
					if face.equiv(w):
						face.visible = True
			Gprime.normalize()
			if check_plane_isomorphism(P, Gprime):
				return True
	return False





	
