"""
Monte Carlo Tree Search over token choices, guided by base LLM logit scores and diffusion critic.
"""
import math
from models.base_llm import BaseLLM
from models.diffusion_critic import DiffusionCritic, add_noise

class MCTSNode:
    def __init__(self, prefix: str, parent=None, token_id: int = None):
        self.prefix = prefix            # text prefix up to this node
        self.parent = parent            # parent MCTSNode
        self.token_id = token_id        # token that led to this node
        self.children = []              # list of child MCTSNodes
        self.visits = 0                 # number of times node was visited
        self.value = 0.0                # total value accrued at node

    def is_leaf(self) -> bool:
        return len(self.children) == 0

class MCTSExplorer:
    def __init__(self,
                 base_llm: BaseLLM,
                 critic: DiffusionCritic,
                 config: dict):
        self.llm = base_llm
        self.critic = critic
        self.K = config.get("K", 5)
        self.num_simulations = config.get("num_simulations", 20)
        self.ucb_c = config.get("ucb_c", 1.4)
        self.noise_level = config.get("noise_level", 0.1)

    def search(self, prefix: str) -> int:
        """
        Runs MCTS starting from `prefix` and returns the best next-token ID.
        """
        root = MCTSNode(prefix)
        # Initialize root visits to avoid log(0)
        root.visits = 1

        for _ in range(self.num_simulations):
            leaf = self._select(root)
            reward = self._simulate(leaf)
            self._backpropagate(leaf, reward)

        # After simulations, pick the child with highest visit count
        if not root.children:
            return None
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.token_id

    def _select(self, node: MCTSNode) -> MCTSNode:
        # Traverse the tree until a leaf
        while not node.is_leaf():
            node = self._uct_select(node)
        # Expand leaf
        self._expand(node)
        # If expanded, return first child for simulation, else return leaf
        return node.children[0] if node.children else node

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        # Upper Confidence Bound selection
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            # If child unvisited, prioritize it
            if child.visits == 0:
                return child
            # Exploitation term
            exploit = child.value / child.visits
            # Exploration term
            explore = self.ucb_c * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node: MCTSNode):
        # Generate top-K next-token candidates
        topk_vals, topk_idx = self.llm.top_k_logits(node.prefix, self.K)
        for tok in topk_idx.tolist():
            new_prefix = node.prefix + self.llm.tokenizer.decode([tok])
            child = MCTSNode(new_prefix, parent=node, token_id=tok)
            node.children.append(child)
        # If no children, this is terminal; else init visits of children to 0

    def _simulate(self, node: MCTSNode) -> float:
        # 1. Get full logits and add noise
        full_logits = self.llm.full_logits(node.prefix)      # [vocab_size]
        noisy = add_noise(full_logits, self.noise_level)     # [vocab_size]
        diff_score = self.critic.compute_score(full_logits, noisy)  # scalar

        # 2. Combine with local logit for each top-K candidate
        topk_vals, _ = self.llm.top_k_logits(node.prefix, self.K)
        combined = [val.item() + diff_score for val in topk_vals]
        return max(combined) if combined else 0.0

    def _backpropagate(self, node: MCTSNode, reward: float):
        # Update value and visits up to the root
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
