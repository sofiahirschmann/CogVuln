from collections import OrderedDict
import copy
import heapq
import itertools
import logging
import math

from psychsim.probability import Distribution
from . import keys

from psychsim.pwl.vector import KeyedVector, VectorDistribution
from psychsim.pwl.matrix import KeyedMatrix
from psychsim.pwl.tree import KeyedTree


class VectorDistributionSet:
    """
    Represents a distribution over independent state vectors, i.e., independent L{VectorDistribution}s
    """
    def __init__(self, start=None):
        self.distributions = {}
        self.certain = {}
        self.keyMap = {}
        if isinstance(start, dict):
            substate = 0
            for key, value in start.items():
                self.join(key, value, substate)
                substate += 1
        elif isinstance(start, VectorDistribution):
            self.distributions[0] = start
            self.keyMap = {k: 0 for k in start.keys()}
        elif start is not None:
            raise TypeError(f'Unknown argument type for initial value: {start.__class__.__name__}')

    def add_distribution(self, dist):
        """
        :param dist: The sub-distribution to add to this state
        :type dist: VectorDistribution
        :warning: Keys in new distribution must not already exist in current distribution
        :return: the index into the newly added distribution
        """
        for key in dist.keys():
            if key in self:
                raise ValueError(f'Cannot add distribution containing {key}, because it already exists')
        substate = 0
        while substate in self.distributions:
            substate += 1
        for key in dist.keys():
            self.keyMap[key] = substate
        self.distributions[substate] = dist
        return substate

    def keys(self):
        return self.keyMap.keys()

    def __contains__(self, key):
        return key in self.keyMap
    
    def __iter__(self):
        """
        Iterate through elements of this set, with each element being a L{VectorDistributionSet} (with probability not necessarily 1)
        """
        size = 1
        domains = {}
        substates = sorted(self.distributions.keys())
        for substate in substates:
            dist = self.distributions[substate]
            domains[substate] = dist.domain()
            size *= len(dist.domain())
        for i in range(size):
            value = self.__class__()
            value.certain.update(self.certain)
            value.keyMap.update(self.keyMap)
            for substate in substates:
                element = domains[substate][i % len(domains[substate])]
                prob = self.distributions[substate][element]
                i //= len(domains[substate])
                value.distributions[substate] = VectorDistribution({element: prob})
            yield value

    def probability(self, vector=None):
        """
        :type vector: KeyedVector
        :return: the probability of the given world according to this state distribution. If no world is given, then return the probability of the overall state
        """
        prob = 1
        if vector is None:
            for distribution in self.distributions.values():
                prob *= distribution.probability()
        else:
            for key,value in vector.items():
                if self.keyMap[key] is not None:
                    prob *= self.marginal(key)[value]
        return prob

    def __len__(self):
        """
        :return: the number of elements in the implied joint distribution
        :rtype: int
        """
        prod = 1
        for dist in self.distributions.values():
            prod *= len(dist)
        return prod

    def __getitem__(self,key):
        return self.marginal(key)
        
    def __setitem__(self,key,value):
        """
        Computes a conditional probability of this distribution given the value for this key. To do so, it removes any elements from the distribution that are inconsistent with the given value and then normalizes.
        .. warning:: If you want to overwrite any existing values for this key use L{join} (which computes a new joint probability)
        """
        self.setitem(key, value)
        
    def setitem(self, key, value):
        substate = self.keyMap[key]
        if substate is None:
            if self.certain[key] != value:
                raise ValueError(f'P({key}={value}) = 0 because {key}={self.certain[key]}')
            else:
                return 1
        else:
            prob = 0
            dist = self.distributions[substate]
            items = dist._Distribution__items
            i = 0
            while i < len(items):
                if math.isclose(items[i][0][key], value):
                    prob += items[i][1]
                    i += 1
                else:
                    del items[i]
            if len(items) == 0:
                raise ValueError(f'P({key}={value}) = 0)')
            else:
                dist.normalize()
            return prob

    def subDistribution(self,key):
        raise DeprecationWarning('Use "subdistribution" instead.')

    def subdistribution(self,key):
        """
        :return: the minimal joint distribution containing this key
        """
        substate = self.keyMap[key]
        if substate is None:
            return VectorDistribution({KeyedVector({key: self.certain[key]}): 1})
        else:
            return self.distributions[substate]

    def __delitem__(self,key):
        """"
        Removes the given column from its corresponding vector (raises KeyError if not present in this distribution)
        """
        substate = self.keyMap[key]
        del self.keyMap[key]
        if substate is None:
            del self.certain[key]
        else:
            dist = self.distributions[substate]
            if len(dist.first()) <= 2:
                # Assume CONSTANT is the other key, so this whole distribution goes
                del self.distributions[substate]
            else:
                # Go through each vector and remove the key
                for vector in dist.domain():
                    prob = dist[vector]
                    del dist[vector]
                    del vector[key]
                    dist.addProb(vector, prob)
                if len(dist) == 1:
                    # No more uncertainty
                    for other_key, value in dist.first().items():
                        self.keyMap[other_key] = None
                        self.certain[other_key] = value
                    del self.distributions[substate]

    def deleteKeys(self,toDelete):
        """
        Removes multiple columns at once
        """
        distributions = {}
        for key in toDelete:
            substate = self.keyMap[key]
            del self.keyMap[key]
            if substate is None:
                del self.certain[key]
            elif substate in distributions:
                old = distributions[substate]
                distributions[substate] = []
                for vector,prob in old:
                    del vector[key]
                    distributions[substate].append((vector,prob))
            else:
                dist = self.distributions[substate]
                distributions[substate] = []
                for vector in dist.domain():
                    prob = dist[vector]
                    del vector[key]
                    distributions[substate].append((vector,prob))
        for substate,dist in distributions.items():
            if len(dist[0][0]) == 1:
                assert next(iter(dist[0][0].keys())) == keys.CONSTANT
                del self.distributions[substate]
            else:
                self.distributions[substate].clear()
                for vector,prob in distributions[substate]:
                    self.distributions[substate].addProb(vector,prob)
            
    def split(self,key):
        """
        :return: partitions this distribution into subsets corresponding to possible values for the given key
        :rtype: dict(str,VectorDistributionSet)
        """
        destination = self.keyMap[key]
        original = self.distributions[destination]
        result = {}
        for vector in original.domain():
            value = vector[key]
            if value not in result:
                # Copy everything from me except the distribution of the given key
                result[value] = self.__class__()
                result.keyMap.update(self.keyMap)
                for substate,distribution in self.distributions.items():
                    if substate != destination:
                        result.distributions[substate] = copy.deepcopy(distribution)
            result[value].distributions[destination][vector] = original[vector]
        return result
        
    def collapse(self, substates, preserve_certainty=True):
        """
        Collapses (in place) the given substates into a single joint L{VectorDistribution}
        """
        if len(substates) > 0:
            if isinstance(next(iter(substates)), str):
                # Why not handle keys, too?
                substates = self.substate(substates)
            if preserve_certainty:
                substates = {s for s in substates
                             if s is not None and len(self.distributions[s]) > 1}
            result = self.merge(substates)
            return result
        else:
            raise ValueError('No substates to collapse')
        
    def uncertain(self):
        """
        :return: C{True} iff this distribution has any uncertainty about the vector
        :rtype: bool
        """
        return sum(map(len, self.distributions.values())) > len(self.distributions)

    def findUncertainty(self, substates=None):
        """
        :param substates: Consider only the given substates as candidates
        :return: a substate containing an uncertain distribution if one exists; otherwise, None
        :rtype: int
        """
        if substates is None:
            substates = self.distributions.keys()
        for substate in substates:
            if substate is not None and len(self.distributions[substate]) > 1:
                return substate
        else:
            return None

    def make_certain(self):
        for substate, dist in list(self.distributions.items()):
            if len(dist) == 1:
                # 100% probability for one vector
                vector = dist.first()
                for var, value in vector.items():
                    if var != keys.CONSTANT:
                        self.keyMap[var] = None
                        self.certain[var] = value
                del self.distributions[substate]

    def vector(self):
        """
        :return: if this distribution contains only a single vector, return that vector; otherwise, throw exception
        :rtype: KeyedVector
        """
        vector = KeyedVector()
        for substate, distribution in self.distributions.items():
            assert len(distribution) == 1, 'Cannot return vector from uncertain distribution'
            vector.update(distribution.domain()[0])
        return vector

    def worlds(self):
        """
        :return: iterator through all possible joint vectors (i.e., possible worlds) and their probabilities
        :rtype: KeyedVector,float
        """
        # Convert to lists now to ensure same ordering throughout
        substates = list(self.distributions.keys())
        domains = {substate: list(self.distributions[substate].domain()) for substate in substates}
        for index in range(len(self)):
            vector = {}
            prob = 1
            for substate in substates:
                subindex = index % len(self.distributions[substate])
                subvector = domains[substate][subindex % len(domains[substate])]
                vector.update(subvector)
                prob *= self.distributions[substate][subvector]
                index = index // len(self.distributions[substate])
            yield KeyedVector(vector), prob

    def select(self, maximize=False, incremental=False):
        """
        Reduce distribution to a single element, sampled according to the given distribution
        :param incremental: if C{True}, then select each key value in series (rather than picking out a joint vector all at once, default is C{False})
        :return: the probability of the selection made
        """
        if incremental:
            prob = KeyedVector()
        else:
            prob = 1
        for substate, distribution in list(self.distributions.items()):
            if incremental:
                prob.update(distribution.select(maximize, incremental))
            else:
                prob *= distribution.select(maximize, incremental)
            # The distribution is now certain, so move variables out of it and delete
            vector = distribution.first()
            for key, value in vector.items():
                self.keyMap[key] = None
                self.certain[key] = value
            del self.distributions[substate]
        return prob

    def substate(self, obj, ignoreCertain=False):
        """
        :return: the substate referred to by all of the keys in the given object
        """
        if isinstance(obj, bool):
            raise DeprecationWarning('If you really need this, please inform management.')
            return set()
        elif ignoreCertain:
            return {self.keyMap[k] for k in obj if k != keys.CONSTANT and 
                    self.keyMap[k] is not None and 
                    len(self.distributions[self.keyMap[k]]) > 1}
        else:
            return {self.keyMap[k] for k in obj if k != keys.CONSTANT}

    def merge(self, substates):
        """
        :return: the substate into which they've all been merged
        """
        destination = None
        for substate in substates:
            if destination is None:
                destination = substate
            elif substate is not None:
                dist = self.distributions[substate]
                self.distributions[destination].merge(dist,True)
                del self.distributions[substate]
                for key in dist.keys():
                    if key != keys.CONSTANT:
                        self.keyMap[key] = destination
        return destination

    def join(self, key, value, substate=None):
        """
        Modifies the distribution over vectors to have the given value for the given key
        :param key: the key to the column to modify
        :type key: str
        :param value: either a single value to apply to all vectors, or else a L{Distribution} over possible values
        :substate: name of substate vector distribution to join with, ignored if the key already exists in this state. By default, find a new substate
        """
        if key in self.keyMap:
            assert substate is None, f'Cannot join {key} to distribution {substate} as it already exists in distribution {self.keyMap[key]}'
            substate = self.keyMap[key]
            if substate is None:
                del self.certain[key]
            else:
                self.distributions[substate].delete_column(key)
        if substate is None:
            if isinstance(value, Distribution):
                substate = 0
                while substate in self.distributions:
                    substate += 1
                self.keyMap[key] = substate
            else:
                # Certain value
                self.keyMap[key] = substate = None
        else:
            self.keyMap[key] = substate
        if substate is None:
            self.certain[key] = value
        else:
            if not substate in self.distributions:
                self.distributions[substate] = VectorDistribution([(KeyedVector({keys.CONSTANT:1}), 1)])
            self.distributions[substate].join(key, value)

    def marginal(self, key):
        substate = self.keyMap[key]
        if substate is None:
            return Distribution({self.certain[key]: 1})
        else:
            return self.distributions[substate].marginal(key)

    def domain(self, key):
        if isinstance(key, str):
            substate = self.keyMap[key]
            if substate is None:
                return [self.certain[key]]
            else:
                return {v[key] for v in self.distributions[substate].domain()}
        elif isinstance(key, list):
            # Identify the relevant subdistributions
            substates = OrderedDict()
            for subkey in key:
                loc = self.keyMap[subkey]
                try:
                    substates[loc].append(subkey)
                    raise RuntimeError('Currently unable to compute domains over interdependent state features')
                except KeyError:
                    substates[loc] = [subkey]
            # Determine the domain of each feature across distributions
            domains = []
            for loc,subkeys in substates.items():
                dist = self.distributions[loc]
                domains.append([[vector[k] for k in subkeys] for vector in dist.domain()])
            return [sum(combo,[]) for combo in itertools.product(*domains)]
        else:
            raise NotImplementedError
    
    def replace(self, substitution, key=None):
        """
        Replaces column values, either across all columns, or only for the specified column
        """
        if key is None:
            for dist in self.distributions.values():
                dist.replace(substitution)
        else:
            self.distributions[self.keyMap[key]].replace(substitution, key)

    def items(self):
        return self.distributions.items()

    def clear(self):
        self.distributions.clear()
        self.keyMap.clear()

    def normalize(self):
        for dist in self.distributions.values():
            dist.normalize()

    def prune(self, threshold):
        raise DeprecationWarning('Use prune_probability or prune_size')

    def prune_probability(self, threshold: float) -> float:
        prob = 1
        for dist in self.distributions.values():
            prob *= dist.prune_probability(threshold)
        return prob

    def prune_size(self, k: int = 1) -> float:
        if len(self) > k:
            # List of substate/distribution pairs for uncertain subdistributions
            dist_list = [dist_tup for dist_tup in self.distributions.items() if len(dist_tup[1]) > 1]
            # Most probable element in each subdistribution (along with index)
            for dist_tup in dist_list:
                dist_tup[1]._Distribution__items.sort(key=lambda tup: (tup[1], tup[0]), reverse=True)
            max_list = [dist_tup[1]._Distribution__items[0] for dist_tup in dist_list]
            # Probability of most probable combination
            max_prob = math.prod([tup[1] for tup in max_list])
            # Worlds added so far
            worlds = []
            heapq.heappush(worlds, (max_prob, [0 for max_item in max_list]))
            for i, dist_tup in enumerate(dist_list):
                current_worlds = worlds[:]
                for element, tup in enumerate(dist_tup[1]._Distribution__items[1:]):
                    # This is not the max item (i.e., it is a viable pruning candidate)
                    obj, prob = tup
                    change = False
                    for other_prob, other_world in current_worlds:
                        if other_world[i] != element+1:
                            # What if we insert this element to this currently max-k world?
                            new_prob = other_prob*prob/dist_tup[1]._Distribution__items[other_world[i]][1]
                            if len(worlds) < k:
                                # Come on in, there's plenty of room
                                new_world = other_world[:]
                                new_world[i] = element+1
                                heapq.heappush(worlds, (new_prob, new_world))
                                change = True
                            elif new_prob > worlds[0][0]:
                                # This new world is better than the least likely world currently active
                                new_world = other_world[:]
                                new_world[i] = element+1
                                heapq.heapreplace(worlds, (new_prob, new_world))
                                change = True
                    if not change:
                        # Worlds only going to get less likely from here
                        break
            target = dist_list[0][0]
            for sub, dist in dist_list[1:]:
                del self.distributions[sub]
            first = True
            items = []
            for prob, world in worlds:
                for i, element in enumerate(world):
                    if i == 0:
                        vector = dist_list[i][1]._Distribution__items[element][0]
                        vector = vector.__class__(vector)
                    else:
                        vector.update(dist_list[i][1]._Distribution__items[element][0])
                if first:
                    for key in vector.keys():
                        self.keyMap[key] = target
                items.append((vector, prob))
            self.distributions[target]._Distribution__items = items
        return self.probability()
            
    def update(self,other,keySet,scale=1):
        # Anyone else mixed up in this?
        toMerge = set(keySet)
        for key in keySet:
            # Any new keys in the same joint as this guy?
            for newKey in self.keyMap:
                if (key in self.keyMap and self.keyMap[key] == self.keyMap[newKey]) \
                   or other.keyMap[key] == other.keyMap[newKey]:
                    # This key is in the same joint
                    if len(self.distributions[self.keyMap[newKey]]) > 1 or \
                       len(other.distributions[other.keyMap[newKey]]) > 1 or \
                        self.marginal(newKey) != other.marginal(newKey):
                        # If there's uncertainty
                        toMerge.add(newKey)
        if len(toMerge) > 0: # If 0, no difference between self and other to begin with
            # Prepare myself to merge
            substates = {self.keyMap[k] for k in toMerge if k in self.keyMap}
            self.collapse(substates,False)
            for key in toMerge:
                if key in self.keyMap:
                    destination = self.keyMap[key]
                    break
            else:
                destination = max(self.keyMap.values())+1
                self.distributions[destination] = VectorDistribution()
            # Align and merge the other
            substates = {other.keyMap[k] for k in toMerge}
            other.collapse(substates,False)
            dist = other.distributions[other.keyMap[key]]
            for vector in dist.domain():
                self.distributions[destination].addProb(vector,dist[vector]*scale)
                for key in vector.keys():
                    if key != keys.CONSTANT:
                        self.keyMap[key] = destination
            return destination
        else:
            return None
                
    def __add__(self,other):
        if isinstance(other,self.__class__):
            assert self.keyMap == other.keyMap,'Currently unable to add distributions with mismatched substates'
            result = self.__class__()
            result.keyMap.update(self.keyMap)
            for substate,value in self.distributions.items():
                result.distributions[substate] = value + other.distributions[substate]
            return result
        else:
            raise NotImplementedError

    def __sub__(self,other):
        if isinstance(other,self.__class__):
            assert self.keyMap == other.keyMap,'Currently unable to subtract distributions with mismatched substates'
            result = self.__class__()
            result.keyMap.update(self.keyMap)
            for substate,value in self.distributions.items():
                result.distributions[substate] = value - other.distributions[substate]
            return result
        else:
            raise NotImplementedError

    def __imul__(self,other,select=False):
        if isinstance(other,KeyedMatrix):
            self.multiply_matrix(other)
        elif isinstance(other,KeyedTree):
            self.multiply_tree(other, select=select)
        elif isinstance(other,KeyedVector):
            self.multiply_vector(other)
        else:
            raise TypeError(f'Unable to multiply state of type {self.__class__.__name__} by {other.__class__.__name__}')
        return self

    def multiply_vector(self, other):
        substates = self.substate(other)
        if substates:
            self.collapse(substates)
            destination = self.findUncertainty(substates)
        else:
            destination = None
        if destination is None:
            destination = len(self.distributions)
            while destination in self.distributions:
                destination -= 1
#                destination = max(self.keyMap.values())+1
        total = 0.
        for key in other:
            if key == keys.CONSTANT and key not in self.keyMap:
                # Assume CONSTANT value is 1?
                total += other[key]
            elif key != keys.CONSTANT and self.keyMap[key] != destination:
                # Certain value for this key
                marginal = self.marginal(key)
                total += other[key]*next(iter(marginal.domain()))
        self.join(keys.VALUE, total, destination)
        dist = self.distributions[destination]
        for index, item in enumerate(dist._Distribution__items):
            item[0][keys.VALUE] += sum([other[key]*item[0].get(key, 0) for key in other])

    def multiply_matrix(self, other):
        # Focus on subset that this matrix affects
        substates = self.substate(other.getKeysIn(), True)
        if substates:
            destination = self.collapse(substates)
        else:
            destination = None
        # Go through each key this matrix sets
        for rowKey, vector in other.items():
            result = Distribution()
            if destination is None:
                # Every value is 100%
                total = 0
                for colKey, weight in vector.items():
                    if colKey == keys.CONSTANT:
                        # Doesn't really matter
                        total += weight
                    else:
                        substate = self.keyMap[colKey]
                        if substate is None:
                            value = self.certain[colKey]
                        else:
                            value = self.distributions[substate].first()[colKey]
                        total += weight*value
#                assert not rowKey in self.keyMap,'%s already exists' % (rowKey)
                destination = len(self.distributions)
                while destination in self.distributions:
                    destination -= 1
                self.join(rowKey, total, destination)
            else:
                # There is at least one uncertain multiplicand
                for state, prob in self.distributions[destination].items():
                    total = 0
                    for colKey, weight in vector.items():
                        if colKey == keys.CONSTANT:
                            # Doesn't really matter
                            total += weight
                        else:
                            substate = self.keyMap[colKey]
                            if substate == destination:
                                value = state[colKey]
                            elif substate is None:
                                value = self.certain[colKey]
                            else:
                                # Certainty
                                value = self.distributions[substate].first()[colKey]
                            total += weight*value
                    state[rowKey] = total
                self.keyMap[rowKey] = destination

    def multiply_tree(self, other, probability=1, select=False):
        if other.isLeaf():
            self *= other.children[None]
        elif other.isProbabilistic():
            if select:
                oldKid, prob = other.children.sample(maximize=select=='max')
                self.multiply_tree(oldKid, probability=prob, select=select)
            else:
                oldKids = list(other.children.domain())
                # Multiply out children, other than first-born
                newKids = []
                for child in oldKids[1:]:
                    prob = other.children[child]
                    assert child.getKeysOut() == oldKids[0].getKeysOut()
                    myChild = copy.deepcopy(self)
                    myChild.multiply_tree(child, probability=prob, select=select)
                    newKids.append(myChild)
                self.multiply_tree(oldKids[0], probability=other.children[oldKids[0]], select=select)
                subkeys = oldKids[0].getKeysOut()
                # Compute first-born child
                newKids.insert(0, self)
                for index in range(len(oldKids)):
                    prob = other.children[oldKids[index]]
                    substates = newKids[index].substate(subkeys)
                    if len(substates) > 1:
                        substate = newKids[index].collapse(substates)
                    else:
                        substate = next(iter(substates))
                    if index == 0:
                        for vector in self.distributions[substate].domain():
                            self.distributions[substate][vector] *= prob
                        mySubstate = substate
                    else:
                        toCollapse = (subkeys,set())
                        while len(toCollapse[0]) + len(toCollapse[1]) > 0:
                            mySubstates = self.substate(toCollapse[1] | \
                                                        set(self.distributions[mySubstate].keys()))
                            if len(mySubstates) > 1:
                                mySubstate = self.collapse(mySubstates,False)
                            else:
                                mySubstate = next(iter(mySubstates))
                            substates = newKids[index].substate(toCollapse[0] | set(newKids[index].distributions[substate].keys()))
                            if len(substates) > 1:
                                substate = newKids[index].collapse(substates, False)
                            else:
                                substate = next(iter(substates))
                            toCollapse = ({k for k in self.distributions[mySubstate].keys() \
                                           if k != keys.CONSTANT and \
                                           not k in newKids[index].distributions[substate].keys()},
                                          {k for k in newKids[index].distributions[substate].keys() \
                                           if k != keys.CONSTANT and \
                                           not k in self.distributions[mySubstate].keys()})
                        distribution = newKids[index].distributions[substate]
                        for vector in distribution.domain():
                            self.distributions[mySubstate].add_prob(vector, distribution[vector]*prob)
        else:
            # Apply the test to this tree
            sufficient = not other.branch.isConjunction  # If any plane test gets this value, no need to test further (e.g., False for conjunctions)
            first = '__null__'
            states = {first: (self, probability, None)}  # (state, probability, substate)
            for p_index, plane in enumerate(other.branch.planes):
                current_states = [(old_value, s_tuple) for old_value, s_tuple in list(states.items()) if old_value != sufficient]
                if len(current_states) == 0:
                    # No more possibility of a different result
                    break
                states = {sufficient: states[sufficient]} if sufficient in states else {}
                for old_value, s_tuple in current_states:
                    s = s_tuple[0]
                    s *= plane[0]
                    should_copy = False
                    valSub = s.keyMap[keys.VALUE]
                    if not math.isclose(s_tuple[1], 1, rel_tol=1e-8):
                        # We've already descended along one side of a branch
                        partials = [substate for substate, dist in self.distributions.items() if not dist.is_complete()]
                        if len(partials) > 1:
                            raise ValueError(f'Miraculous but incorrect appearance of multiple subdistributions with probability mass < 1 {[self.distributions[s].probability() for s in partials]}')
                        elif len(partials) == 0:
                            raise ValueError('Where did all the incompleteness go?')
                        if partials[0] != valSub:
                            # The test result covers a different set of variables than was tested upstream
                            valSub = s.merge([partials[0], valSub])
                    del s.keyMap[keys.VALUE]
                    # Iterate through possible test results
                    vector_list = list(s.distributions[valSub].items())
                    s.distributions[valSub].clear()
                    for vector, prob in vector_list:
                        # Test this vector against the hyperplane
                        test = other.branch.evaluate(vector[keys.VALUE], p_index)
                        del vector[keys.VALUE]
                        if test not in other.children:
                            # There may not be a branch for this value, 
                            # because we determined it has 0 probability.
                            # Or maybe we just forgot.
                            if test is None:
                                logging.warning('Missing fallback branch in tree:\n%s' % (str(other)))
                            else:
                                logging.warning('Missing branch for value %s in tree:\n%s' % (test, str(other)))
                            continue
                        if test in states:
                            if len(vector) > 1:
                                # Merge in this vector's keys with an existing matching test result
                                new_sub = states[test][0].merge(states[test][0].substate(vector.keys()))
                            else:
                                # Nothing to merge, just carry over the substate from an existing matching result
                                new_sub = states[test][2]
                            old_dist = states[test][0].distributions[states[test][2]]
                            for old_key in old_dist.keys():
                                if old_key not in vector:
                                    raise RuntimeError
                                    sub_dist = s[old_key]
                                    if len(sub_dist) > 1:
                                        raise ValueError('Worlds are branching in a way that I am not prepared to handle')
                                    vector[old_key] = sub_dist.first()
                            if s is self and not should_copy:
                                for old_vec, old_prob in states[test][0].distributions[new_sub].items():
                                    s.distributions[valSub].addProb(old_vec, old_prob)
                                states[test] = (self, states[test][1]+prob, valSub) 
                                should_copy = True
                            else:
                                states[test] = (states[test][0], states[test][1]+prob, states[test][2])
                            if len(vector) > 1:
                                assert states[test][0].distributions[states[test][2]].addProb(vector, prob) <= 1
                        elif should_copy:
                            states[test] = (copy.deepcopy(s), prob, valSub)
                            states[test][0].distributions[valSub].clear()
                            assert states[test][0].distributions[valSub].addProb(vector, prob) <= 1
                        else:
                            states[test] = (s, prob, valSub)
                            should_copy = True
                            assert states[test][0].distributions[valSub].addProb(vector, prob) <= 1
                        if states[test][0] is self:
                            first = test
                    if len(s.distributions[valSub]) == 0:
                        del s.distributions[valSub]
            assert states, 'Empty result of multiplication'
            self.multiply_tree(other.children[first], probability*states[first][1], select)
            del states[first]
            new_keys = set(other.getKeysOut())
            for test, s_plus in states.items():
                s = s_plus[0]
                branch_keys = set(s.distributions[s_plus[2]].keys()) - {keys.CONSTANT}
                s.multiply_tree(other.children[test], states[test][1], select)
                self.update(s, new_keys | branch_keys)
            for k in new_keys:
                for el, p in self.marginal(k).items():
                    if p-1e-8 > 1:
                        raise ValueError(f'Marginal probability of {k} is > 1')

    def __rmul__(self,other):
        if isinstance(other,KeyedVector) or isinstance(other,KeyedTree):
            self *= other
            substate = self.keyMap[keys.VALUE]
            distribution = self.distributions[substate]
            del self.keyMap[keys.VALUE]
            total = 0.
            for vector, prob in distribution.items():
                total += prob*vector[keys.VALUE]
                del vector[keys.VALUE]
            if len(vector) <= 1:
                del self.distributions[substate]
#            for s in self.distributions:
#                assert s in self.keyMap.values(),self.distributions[s]
#            for k,s in self.keyMap.items():
#                if k != keys.CONSTANT:
#                    assert s in self.distributions
            return total
        else:
            raise NotImplementedError

    def rollback(self, debug=False):
        """
        Removes any current state values and makes any future state values the current ones
        :param debug: if True, then run some checks on the values
        """
        # What keys have both current and future values?
        pairs = [k for k in self.keyMap if k != keys.CONSTANT and
                 not keys.isFuture(k) and keys.makeFuture(k) in self.keyMap]
        for now in pairs:
            nowSub = self.keyMap[now]
            future = keys.makeFuture(now)
            futureSub = self.keyMap[future]
            del self.keyMap[future]
            if nowSub is None:
                if futureSub is None:
                    self.certain[now] = self.certain[future]
                    del self.certain[future]
                else:
                    del self.certain[now]
                distribution = None
            else:
                distribution = self.distributions[nowSub]
                items = list(distribution.items())
                distribution.clear()
                for vector, prob in items:
                    if nowSub == futureSub:
                        # Kill two birds with one stone
                        vector[now] = vector[future]
                        del vector[future]
                    else:
                        del vector[now]
                    if len(vector) > 1:
                        distribution.add_prob(vector, prob)
                    elif len(vector) == 1 and debug:
                        assert next(iter(vector.keys())) == keys.CONSTANT
            if nowSub != futureSub:
                # Kill two birds with two stones
                if distribution is not None and len(distribution) == 0:
                    del self.distributions[nowSub]
                self.keyMap[now] = futureSub
                if futureSub is None:
                    self.certain[now] = self.certain[future]
                else:
                    distribution = self.distributions[futureSub]
                    items = list(distribution.items())
                    distribution.clear()
                    for vector, prob in items:
                        vector[now] = vector[future]
                        del vector[future]
                        distribution.add_prob(vector, prob)
            if debug:
                assert now in self.keyMap
                assert self.keyMap[now] in self.distributions,now
        if debug:
            for s in self.distributions:
                assert s in self.keyMap.values(),'Distribution %s is missing\n%s' % (s,self.distributions[s])
                for k in self.distributions[s].keys():
                    assert not keys.isFuture(k),'Future key %s persists after rollback' \
                        % (k)
            for k,s in self.keyMap.items():
                if k != keys.CONSTANT:
                    assert s in self.distributions,'%s: %s' % (k,s)
                assert not keys.isFuture(k)

    def simpleRollback(self,futures):
        # Make the future the present
        for key in futures:
            future = keys.makeFuture(key)
            oldstate = self.keyMap[key]
            newstate = self.keyMap[future]
            dist = self.distributions[newstate]
            if oldstate == newstate:
                for vector in dist.domain():
                    prob = dist[vector]
                    del dist[vector]
                    vector[key] = vector[future]
                    del vector[future]
                    dist.addProb(vector,prob)
            elif len(dist) > 1:
                # New value is probabilistic, not a single value, so update old value across possible worlds
                for vector in dist.domain():
                    prob = dist[vector]
                    del dist[vector]
                    vector[key] = vector[future]
                    del vector[future]
                    dist[vector] = prob
                self.keyMap[key] = newstate
                # Remove old state values
                dist = self.distributions[oldstate]
                if len(dist.first()) > 2:
                    # Other variables still remain
                    for vector in dist.domain():
                        prob = dist[vector]
                        del dist[vector]
                        del vector[key]
                        dist.addProb(vector,prob)
                else:
                    del self.distributions[oldstate]
            else:
                vector = dist.first()
                value = vector[future]
                del dist[vector]
                del vector[future]
                if len(vector) > 1:
                    dist[vector] = 1
                else:
                    del self.distributions[newstate]
                dist = self.distributions[oldstate]
                for vector in dist.domain():
                    prob = dist[vector]
                    del dist[vector]
                    vector[key] = value
                    dist.addProb(vector,prob)
            del self.keyMap[future]

    def __eq__(self,other):
        if not isinstance(other, VectorDistributionSet):
            return False
        remaining = set(self.keyMap.keys())
        if remaining != set(other.keyMap.keys()):
            # The two do not even contain the same columns
            return False
        else:
            while remaining:
                key = remaining.pop()
                if self.keyMap[key] is None:
                    if other.keyMap[key] is None:
                        if self.certain[key] != other.certain[key]:
                            return False
                    else:
                        return False
                elif other.keyMap[key] is None:
                    return False
                else:
                    distributionMe = self.distributions[self.keyMap[key]]
                    distributionYou = other.distributions[other.keyMap[key]]
                    if distributionMe != distributionYou:
                        return False
                    remaining -= set(distributionMe.keys())
            return True

    def delete_value(self, key, value):
        """Removes the given value for the given key from the state and then renormalizes
        :param value: value (or set of values) to be removed
        """
        distribution = self.distributions[self.keyMap[key]]
        for vector in distribution.domain():
            if isinstance(value, set):
                if vector[key] in value:
                    del distribution[vector]
            elif vector[key] == value:
                del distribution[vector]
        distribution.normalize()

    def __deepcopy__(self,memo):
        result = self.__class__()
        for substate, distribution in self.distributions.items():
            new = copy.deepcopy(distribution)
            result.distributions[substate] = new
        result.keyMap.update(self.keyMap)
        result.certain.update(self.certain)
        return result
    
    def __str__(self):
        uncertain = '\n---\n'.join([str(dist) for dist in self.distributions.values()])
        if self.certain:
            vector = KeyedVector(self.certain)
            if uncertain:
                return f'100%\n{vector.sortedString()}\n---\n{uncertain}'
            else:
                return f'100%\n{vector.sortedString()}'
        else:
            return uncertain

    def copySubset(self, ignore=None, include=None):
        raise DeprecationWarning('Use copy_subset instead')

    def copy_subset(self, ignore=None, include=None):
        result = self.__class__()
        if ignore is None and include is None:
            # Ignoring nothing, including everything, so this is just a copy
            return self.__deepcopy__({})
        if include is None:
            include = set(self.keys())
        if ignore is None:
            keySubset = include
        else:
            keySubset = include - ignore
        for key in keySubset:
            if key not in result and key in self:
                if self.keyMap[key] is None:
                    result.certain[key] = self.certain[key]
                    result.keyMap[key] = None
                else:
                    distribution = self.distributions[self.keyMap[key]]
                    substate = len(result.distributions)
                    result.distributions[substate] = distribution.__class__()
                    intersection = distribution.keys() & keySubset #[k for k in distribution.keys() if k in keySubset]
                    for subkey in intersection:
                        result.keyMap[subkey] = substate
                    new_dist = []
                    for vector, prob in distribution.items():
                        new_dict = {subkey: vector[subkey] for subkey in intersection}
                        new_dict[keys.CONSTANT] = 1
                        new_dist.append((vector.__class__(new_dict), prob))
                    result.distributions[substate] = VectorDistribution(new_dist)
                    result.distributions[substate].remove_duplicates()
        return result
                    
    def verifyIntegrity(self, sumToOne=False):
        for key in self.keys():
            assert self.keyMap[key] in self.distributions,'Distribution %s missing for key %s' % \
                (self.keyMap[key],key)
            distribution = self.distributions[self.keyMap[key]]
            for vector in distribution.domain():
                assert key in vector,'Key %s is missing from vector\n%s\nProb: %d%%' % \
                    (key,vector,distribution[vector]*100)
                for other in vector:
                    assert other == keys.CONSTANT or self.keyMap[other] == self.keyMap[key] ,\
                        f'Unmapped key {other} is in vector\n{vector}'
            if sumToOne:
                assert (distribution.probability()-1)<.000001,'Distribution sums to %4.2f' % (distribution.probability())
            else:
                assert distribution.probability()<1.000001, f'Distribution sums to {distribution.probability()}'

    def copy_value(self, old_key, new_key):
        """
        Modifies the state so that the distribution over the new key's values is identical to that of the old key
        """
        substate = self.keyMap[old_key]
        self.keyMap[new_key] = substate
        if substate is None:
            self.certain[new_key] = self.certain[old_key]
        else:
            dist = self.distributions[substate]
            for vector, prob in list(dist.items()):
                del dist[vector]
                vector[new_key] = vector[old_key]
                dist[vector] = prob

    def is_minimal(self):
        """
        :return: False iff any non-singleton distributions are non-singleton for all variables in that distribution"
        """
        for dist in self.distributions.values():
            if len(dist) > 1:
                for vector in dist.domain():
                    for key in vector.keys():
                        if key != keys.CONSTANT and len(self.marginal(key)) == 1:
                            # This feature is 100% certain, yet exists in a distribution that is uncertain
                            return False
        else:
            return True

    def diff(self, other):
        """
        :return: a dictionary of differences between me and the given state
        """
        result = {'only_me': self.keys() - other.keys(),
                  'only_you': other.keys() - self.keys(),
                  'dependency mismatch': {},
                  'probability mismatch': set()}
        for key in self.keys() & other.keys():
            if self.marginal(key) != other.marginal(key):
                result['probability mismatch'].add(key)
            dist_me = self.distributions[self.keyMap[key]]
            dist_you = other.distributions[other.keyMap[key]]
            match = dist_me.keys() & dist_you.keys()
            mismatch = (dist_me.keys() | dist_you.keys()) - match
            if mismatch:
                result['dependency mismatch'][key] = mismatch
        return result

    def max_len(self):
        """
        :return: the size of the largest subdistribution
        """
        return max([len(dist) for dist in self.distributions.values()])
        