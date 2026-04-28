import numpy as np

from typing import Tuple
from itertools import chain
from scipy.optimize import minimize

from src.utils.variables import Variables
from src.fca.lattice import cover_relations

class ForceDirectedPlacement():
    '''
    Optimize the layout of a Concept Lattice using a force-directed approach.
    This class minimizes an energy function composed of repulsive, attractive, 
    and gravitational forces to improve diagram readability.

    Attributes
    ----------
    vars : Variables
        The container holding context data and layout parameters
    coordinates : Dict[int, np.ndarray]
        Mapping of concept IDs to their optimized 2D coordinates
    e_rep : float
        The repulsive energy of the system
    e_att : float
        The attractive energy of the system
    e_grav : float
        The gravitational energy of the system
    final_forces : Dict[str, np.ndarray]
        The decomposed force vectors used for visualization and analysis
    '''
    def __init__(self, variables: Variables):
        '''
        Initialize and run the optimization process.

        Parameters
        ----------
        variables : Variables
            The container class holding the data
        '''
        self.vars = variables
        self.coordinates = dict({})

        self._optimize_layout()
        self._final_force()

    def _optimize_layout(self):
        '''
        Compute an optimized layout by minimizing the total energy function
        using the Conjugate Gradient (CG) method.
        '''
        res = minimize(
            fun=self._total_energy_and_gradient,
            x0=np.array([self.vars.base_vectors[v] for v in self.vars.elements]).flatten(),
            method='CG',
            jac=True,
            options={'maxiter': 1000}
        )

        # store optimized vectors
        optimized_matrix = res.x.reshape(-1, 2)
        for i, v in enumerate(self.vars.elements):
            self.vars.base_vectors[v] = optimized_matrix[i]

        # derive coordinates for optimized layout
        self.base_vectors = self.vars.base_vectors
        self.coordinates = {}
        for c in self.vars.concepts:
            self.coordinates[c] = self._concept_pos(c, np.array([self.vars.base_vectors[v] for v in self.vars.elements]))

    def _total_energy_and_gradient(self, flat_vectors: np.ndarray) -> Tuple[float, np.ndarray]:
        '''
        Compute the total energy of forces in the drawing of the Concept Lattice.
        1. Repulsive Energy (E_rep):
            Maximizes distance between nodes and non-incident edges.
        2. Attractive Energy (E_att):
            Minimizes edge lengths to keep related concepts close.
        3. Gravitational Energy (E_grav): 
            Constraints vectors to safe angles to ensure an upward-directed, readable diagram.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of object and attribute vectors.

        Returns
        -------
        energy : float
            The total energy of the three forces.
        '''
        # forces
        self.e_rep, self.gradients_rep = self._repulsive_force(flat_vectors)
        self.e_att, self.gradients_att = self._attractive_force(flat_vectors)
        self.e_grav, self.gradients_grav = self._gravitational_force(flat_vectors)

        # weights
        self.energy = self.vars.w_rep * self.e_rep + self.vars.w_att * self.e_att + self.vars.w_grav * self.e_grav
        self.gradients = self.vars.w_rep * self.gradients_rep + self.vars.w_att * self.gradients_att + self.vars.w_grav * self.gradients_grav

        # conjugate gradient (expects a negative gradient)
        return self.energy, (self.gradients*-1).flatten()
    
    def _concept_pos(self, concept, vectors) -> np.ndarray:
        '''
        Compute concept position based on present vectors.

        Parameters
        ----------
        concept : int
            Concept to compute the position for
        vectors : Dict[int, np.array]
            Dictionary assigning vectors to objects and attributes  

        Returns
        -------
        position : np.array
            Position of the concept
        '''
        base_vectors = np.array([self.vars.element_map[v] for v in (self.vars.extents[concept] | self.vars.intents[concept])])

        # (0, 0) if concept has no objects and no attributes
        if not base_vectors.size:
            return np.zeros(2)
        
        return np.sum(vectors[base_vectors], axis=0)

    def _repulsive_force(self, flat_vectors) -> Tuple[float, np.ndarray]:
        '''
        Compute the Repulsive Energy, which maximizes the distance between nodes and non-incident edges.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of ojbect and attribute vectors.

        Returns
        -------
        energy : float
            The repulsive energy
        gradient : np.ndarray
            Gradient of the repulsive energy
        '''
        vectors = flat_vectors.reshape(-1, 2)
        positions = [
            self._concept_pos(c, vectors)
            for c in self.vars.concepts
        ]

        e_rep = 0.0
        gradients_rep = np.zeros_like(vectors)

        for v, w in enumerate(positions):
            for (i, j) in cover_relations(self.vars.lattice):
                # edges without concept c
                if v == i or v == j:
                    continue

                # edge v_1, v_2 with v_1 below v_2
                if self.vars.extents[i] <= self.vars.extents[j]:
                    v_1, v_2 = i, j
                else:
                    v_1, v_2 = j, i
                
                w_1, w_2 = positions[v_1], positions[v_2]

                extent_v, intent_v = self.vars.extents[v], self.vars.intents[v]
                extent_v_1, intent_v_1 = self.vars.extents[v_1], self.vars.intents[v_1]
                extent_v_2, intent_v_2 = self.vars.extents[v_2], self.vars.intents[v_2]

                for n_i in chain(self.vars.objects, self.vars.attributes):
                    F_G = {i: False for i in range(1, 9)}
                    F_M = {i: False for i in range(1, 9)}

                    # object distribution
                    if n_i in self.vars.extents[0]:
                        F_G[1] = (n_i not in extent_v) and (n_i not in extent_v_1) and (n_i not in extent_v_2)
                        F_G[2] = (n_i not in extent_v) and (n_i not in extent_v_1) and (n_i in extent_v_2)
                        F_G[3] = (n_i not in extent_v) and (n_i in extent_v_1) and (n_i not in extent_v_2)
                        F_G[4] = (n_i not in extent_v) and (n_i in extent_v_1) and (n_i in extent_v_2)
                        F_G[5] = (n_i in extent_v) and (n_i not in extent_v_1) and (n_i not in extent_v_2)
                        F_G[6] = (n_i in extent_v) and (n_i not in extent_v_1) and (n_i in extent_v_2)
                        F_G[7] = (n_i in extent_v) and (n_i in extent_v_1) and (n_i not in extent_v_2)
                        F_G[8] = (n_i in extent_v) and (n_i in extent_v_1) and (n_i in extent_v_2)

                    # attribute distribution
                    else:
                        F_M[1] = (n_i not in intent_v) and (n_i not in intent_v_1) and (n_i not in intent_v_2)
                        F_M[2] = (n_i not in intent_v) and (n_i not in intent_v_1) and (n_i in intent_v_2)
                        F_M[3] = (n_i not in intent_v) and (n_i in intent_v_1) and (n_i not in intent_v_2)
                        F_M[4] = (n_i not in intent_v) and (n_i in intent_v_1) and (n_i in intent_v_2)
                        F_M[5] = (n_i in intent_v) and (n_i not in intent_v_1) and (n_i not in intent_v_2)
                        F_M[6] = (n_i in intent_v) and (n_i not in intent_v_1) and (n_i in intent_v_2)
                        F_M[7] = (n_i in intent_v) and (n_i in intent_v_1) and (n_i not in intent_v_2)
                        F_M[8] = (n_i in intent_v) and (n_i in intent_v_1) and (n_i in intent_v_2)

                    # not possible by order relation
                    if F_G[3] or F_G[7] or F_M[2] or F_M[6]:
                        continue

                    # would not change conflict distance
                    if F_G[1] or F_G[8] or F_M[1] or F_M[8]:
                        continue
                    
                    ##################################################
                    # case 1:
                    # (w_1 - w) \cdot (w_2 - w_1) > 0
                    # concept v lies below v_1
                    ##################################################
                    if np.dot(w_1 - w, w_2 - w_1) > 0:
                        # |w_1 - w|
                        dist = np.linalg.norm(w_1 - w)

                        ##############################################
                        # F_{G_4} = F_{M_3} = F_{M_4}
                        ##############################################
                        if F_G[4] or F_M[3] or F_M[4]:
                            # (1 / d(w, f)^2) * e(w_1 - w)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * ((w_1 - w) / dist)

                        ##############################################
                        # F_{G_5} = F_{G_6} = F_{M_5}
                        ##############################################
                        elif F_G[5] or F_G[6] or F_M[5]: 
                            # (1 / d(w, f)^2) * e(w - w_1)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * ((w - w_1) / dist)

                    ##################################################
                    # case 2:
                    # (w_2 - w) \cdot (w_2 - w_1) < 0
                    # concept v lies above v_2
                    ##################################################
                    elif np.dot(w_2 - w, w_2 - w_1) < 0:
                        # |w_2 - w|
                        dist = np.linalg.norm(w_2 - w)

                        ##############################################
                        # F_{G_2} = F_{G_4} = F_{M_4}
                        ##############################################
                        if F_G[2] or F_G[4] or F_M[4]:
                            # (1 / d(w, f)^2) * e(w_2 - w)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * ((w_2 - w) / dist)

                        ##############################################
                        # F_{G_5} = F_{M_5} = F_{M_7}
                        ##############################################
                        elif F_G[5] or F_M[5] or F_M[7]:
                            # (1 / d(w, f)^2) * e(w - w_2)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * ((w - w_2) / dist)

                    ##################################################
                    # case 3:
                    # (w_2 - w_1) \cdot (w - w_2) \leq 0, (w - w_1) \cdot (w - w_2) \geq 0
                    # concept w lies above w_1 and below w_2 
                    ##################################################
                    else:
                        # perpendicular distance
                        A = np.abs(np.cross(w_1 - w, w_2 - w))
                        f = w_2 - w_1
                        dist = np.maximum(A / np.linalg.norm(f), 1e-3)
                        
                        # (w_1 - w) \times (w_2 - w) \geq 0 -> w lies left of w_1w_2
                        # (w_1 - w) \times (w_2 - w) < 0 -> w lies right of w_1w_2
                        l = 1 if np.cross(w_1 - w, w_2 - w) >= 0 else -1

                        # n_+(f)
                        x_f, y_f = f
                        n_plus_f = np.array([-y_f, x_f])
                        h = np.maximum(A / np.linalg.norm(f), 1e-3)

                        ##############################################
                        # F_{G_2}
                        ##############################################
                        if F_G[2]:
                            # (1 / d(w, f)^2) * -sqrt(((w_1 - w)^2 - |h|^2) / |f|^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * -np.sqrt(abs((np.linalg.norm(w_1 - w)**2 - h**2)) / np.linalg.norm(f)**2) * ((n_plus_f * l) / np.linalg.norm(f))

                        ##############################################
                        # F_{M_3}
                        ##############################################
                        elif F_M[3]:
                            # (1 / d(w, f)^2) * -sqrt(((w_2 - w)^2 - |h|^2) / |f|^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * -np.sqrt(abs((np.linalg.norm(w_2 - w)**2 - h**2)) / np.linalg.norm(f)**2) * ((n_plus_f * l) / np.linalg.norm(f))

                        ##############################################
                        # F_{G_4} = F_{M_4}
                        ##############################################
                        elif F_G[4] or F_M[4]:
                            # (1 / d(w, f)^2) * -((n_+(f) * l) / |f|)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * -((n_plus_f * l) / np.linalg.norm(f))

                        ##############################################
                        # F_{G_5} = F_{M_5}
                        ##############################################
                        elif F_G[5] or F_M[5]:
                            # (1 / d(w, f)^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * (n_plus_f * l) / np.linalg.norm(f)

                        ##############################################
                        # F_{G_6}
                        ##############################################
                        elif F_G[6]:
                            # (1 / d(w, f)^2) * sqrt(((w_2 - w)^2 - |h|^2) / |f|^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * np.sqrt(abs((np.linalg.norm(w_2 - w)**2 - h**2)) / np.linalg.norm(f)**2) * ((n_plus_f * l) / np.linalg.norm(f))

                        ##############################################
                        # F_{M_7}
                        ##############################################
                        elif F_M[7]:
                            # (1 / d(w, f)^2) * sqrt(((w_1 - w)^2 - |h|^2) / |f|^2) * ((n_+(f) * l) / |f|)
                            gradients_rep[self.vars.element_map[n_i]] += (1 / dist**2) * np.sqrt(abs((np.linalg.norm(w_1 - w)**2 - h**2)) / np.linalg.norm(f)**2) * ((n_plus_f * l) / np.linalg.norm(f))

                # 1 / d(w, f)
                e_rep += 1.0 / dist
        
        return e_rep, gradients_rep


    def _attractive_force(self, flat_vectors) -> Tuple[float, np.ndarray]:
        '''
        Compute the Attractive Energy, which minimizes edge lengths to keep related concepts close.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of ojbect and attribute vectors.

        Returns
        -------
        energy : float
            The attractive energy
        gradient : np.ndarray
            Gradient of the attractive energy
        '''
        vectors = flat_vectors.reshape(-1, 2)
        positions = [
            self._concept_pos(c, vectors)
            for c in self.vars.concepts
        ]

        e_att = 0.0
        gradients_att = np.zeros_like(vectors)

        for (i, j) in cover_relations(self.vars.lattice):
            # edge v_1, v_2 with v_1 below v_2
            if self.vars.extents[i] <= self.vars.extents[j]:
                v_1, v_2 = i, j
            else:
                v_1, v_2 = j, i
            
            w_1, w_2 = positions[v_1], positions[v_2]

            extent_v_1, intent_v_1 = self.vars.extents[v_1], self.vars.intents[v_1]
            extent_v_2, intent_v_2 = self.vars.extents[v_2], self.vars.intents[v_2]

            # m \in v_1 \ v_2
            for m_i in intent_v_1 - intent_v_2:
                # 2 * (w_2 - w_1)
                gradients_att[self.vars.element_map[m_i]] += 2 * (w_2 - w_1)

            # g in v_2 \ v_1
            for g_i in extent_v_2 - extent_v_1:
                gradients_att[self.vars.element_map[g_i]] += 2 * (w_1 - w_2)

            # |w_2 - w_1|^2
            e_att += np.sum((w_2 - w_1)**2)
        
        return e_att, gradients_att

    def _gravitational_force(self, flat_vectors) -> Tuple[float, np.ndarray]:
        '''
        Compute the Gravitational Energy, which constraints all vectors to safe angles
        to ensure an upward-directed, readable diagram. Object vectors upward-directed
        and attribute vectors downward-directed.

        Parameters
        ----------
        flat_vectors : np.array
            A 1D flattened array of ojbect and attribute vectors.

        Returns
        -------
        energy : float
            The gravitational energy
        gradient : np.ndarray
            Gradient of the gravitational energy
        '''
        vectors = flat_vectors.reshape(-1, 2)

        # angle phi_0
        phi_0_G = np.pi / (self.vars.N_g + 1)
        phi_0_M = np.pi / (self.vars.N_m + 1)

        # integration constants
        E_0 = -phi_0_G - np.sin(phi_0_G) * np.cos(phi_0_G)
        E_1 = E_0 + np.pi
        E_2 = phi_0_M + np.sin(phi_0_M) * np.cos(phi_0_M)
        E_3 = E_2 - np.pi

        e_grav = 0.0
        gradients_grav = np.zeros_like(vectors)

        for n_i, (x, y) in enumerate(vectors):
            if abs(y) < 1e-15:
                continue
            # object vector
            if n_i < self.vars.N_g:
                phi_n_i = np.arctan2(y, x)
                if abs(phi_n_i) <= 1e-15:
                    phi_n_i = 1e-3
                # flat vector
                if (0 <= phi_n_i <= phi_0_G) or ((np.pi - phi_0_G) <= phi_n_i <= np.pi):
                    # 1, if 0 < phi_n_i < phi_0_G
                    # -1, if phi_0_G < phi_n_i < pi
                    direction = 1 if 0 < phi_n_i < phi_0_G else -1

                    # angle too flat on the right side
                    if 0 <= phi_n_i <= phi_0_G:
                        # E_grav(n_i) = phi_n_i + cot(phi_n_i) sin(phi_0_G)^2 + E_0
                        e_grav += phi_n_i + (np.cos(phi_n_i) / (np.sin(phi_n_i))) * (np.sin(phi_0_G)**2) + E_0

                    # angle too flat on the left side
                    elif (np.pi - phi_0_G) <= phi_n_i <= np.pi:
                        # E_grav(m) = -phi_n_i - cot(phi_n_i) sin(phi_0)^2 + E_1
                        e_grav += -phi_n_i - (np.cos(phi_n_i) / (np.sin(phi_n_i))) * (np.sin(phi_0_G)**2) + E_1

                    # n_-(n_i) * ((sin(phi_n_i)^2 - sin(phi_0_G)^2) / y(n_i)^2) * direction
                    gradients_grav[n_i] += np.array([y, -x]) * ((np.sin(phi_n_i)**2 - np.sin(phi_0_G)**2) / y**2) * direction

                # wrong direction (downwards)
                elif phi_n_i < 0:
                    penalty = 1e3
                    # linear penalty based on how far below the axis it is
                    e_grav += penalty * y**2
                    # derivative of energy
                    gradients_grav[n_i, 0] = 0.0
                    gradients_grav[n_i, 1] = penalty * -2 * y

            # attribute vector
            else:
                phi_n_i = np.arctan2(y, x)
                if abs(phi_n_i) <= 1e-15:
                    phi_n_i = -1e-3

                # flat vector
                if (-phi_0_M <= phi_n_i <= 0) or (-np.pi <= phi_n_i <= (-np.pi + phi_0_M)):
                    # 1, if -pi < phi_n_i < -phi_0_M
                    # -1, if -phi_0_M < phi_n_i < 0
                    direction = 1 if -np.pi < phi_n_i < -phi_0_M else -1

                    # angle too flat on the right side
                    if -phi_0_M <= phi_n_i <= 0:
                        # E_grav(n_i) = phi_n_i + cot(phi_n_i) sin(phi_0_M)^2 + E_2
                        e_grav += phi_n_i + (np.cos(phi_n_i) / (np.sin(phi_n_i))) * (np.sin(phi_0_M)**2) + E_2

                    # angle too flat on the left side
                    elif -np.pi <= phi_n_i <= (-np.pi + phi_0_M):
                        # E_grav(m) = -phi_n_i - cot(phi_n_i) sin(phi_0)^2 + E_3
                        e_grav += -phi_n_i - (np.cos(phi_n_i) / (np.sin(phi_n_i))) * (np.sin(phi_0_M)**2) + E_3

                    # n_-(n_i) * ((sin(phi_n_i)^2 - sin(phi_0_M)^2) / y(n_i)^2) * direction
                    gradients_grav[n_i] += np.array([y, -x]) * ((np.sin(phi_n_i)**2 - np.sin(phi_0_M)**2) / y**2) * direction

                # wrong direction (upwards)
                elif phi_n_i > 0:
                    penalty = 1e3
                    # linear penalty based on how far below the axis it is
                    e_grav += penalty * y**2
                    # derivative of energy
                    gradients_grav[n_i, 0] = 0.0
                    gradients_grav[n_i, 1] = penalty * -2 * y

        return e_grav, gradients_grav
    
    def _final_force(self):
        '''
        Calculate the resulting decomposed forces for each concept 
        at the end of the optimization. This is primarily for visualization.
        '''
        self.final_forces = {
            'G_rep': np.zeros((self.vars.N_c, 2)), 'M_rep': np.zeros((self.vars.N_c, 2)),
            'G_att': np.zeros((self.vars.N_c, 2)), 'M_att': np.zeros((self.vars.N_c, 2)),
            'G_grav': np.zeros((self.vars.N_c, 2)), 'M_grav': np.zeros((self.vars.N_c, 2))
        }

        for c in self.vars.concepts:
            for g in self.vars.extents[c]:
                self.final_forces['G_rep'][c] += self.gradients_rep[self.vars.element_map[g]] * self.vars.w_rep
                self.final_forces['G_att'][c] += self.gradients_att[self.vars.element_map[g]] * self.vars.w_att
                self.final_forces['G_grav'][c] += self.gradients_grav[self.vars.element_map[g]] * self.vars.w_grav
        
            for m in self.vars.intents[c]:
                self.final_forces['M_rep'][c] += self.gradients_rep[self.vars.element_map[m]] * self.vars.w_rep
                self.final_forces['M_att'][c] += self.gradients_att[self.vars.element_map[m]] * self.vars.w_att
                self.final_forces['M_grav'][c] += self.gradients_grav[self.vars.element_map[m]] * self.vars.w_grav
