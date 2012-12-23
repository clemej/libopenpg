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
	pass
