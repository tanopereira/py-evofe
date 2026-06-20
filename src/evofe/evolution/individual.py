import random
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from ..builtin import evo_transformers

def create_gene(transformer_name: str, input_cols: List[str]) -> 'Gene':
    params = {}
    if transformer_name in ["pca", "truncated_svd", "umap"]:
        C = max(2, int(round(np.log2(len(input_cols)))))
        params["comp_idx"] = random.randint(0, C - 1)
    elif transformer_name in ["genie_centroid_dist", "lumbermark_centroid_dist"]:
        k = random.randint(2, 5)
        params["k"] = k
        comp = random.randint(0, k - 1)
        params["comp_idx"] = comp
        params["centroid_idx"] = comp
        if transformer_name == "genie_centroid_dist":
            params["gini_threshold"] = round(random.uniform(0.1, 0.9), 2)
    elif transformer_name == "one_hot_encode":
        params["comp_idx"] = random.randint(1, 6)
    elif transformer_name == "genie":
        params["k"] = random.randint(2, 5)
        params["gini_threshold"] = round(random.uniform(0.1, 0.9), 2)
    elif transformer_name == "lumbermark":
        params["k"] = random.randint(2, 5)
    elif transformer_name in ["quantile_binning", "quantile_binning_cat"]:
        params["Q"] = random.randint(3, 10)
    elif transformer_name in ["log_binning", "log_binning_cat"]:
        params["base"] = random.randint(2, 10)
    elif transformer_name == "target_encode_multiclass":
        params["comp_idx"] = random.randint(0, 4)
    elif transformer_name == "datetime_extract":
        params["component"] = random.choice(["year", "month", "day", "hour", "day_of_week", "weekend"])
    elif transformer_name == "power":
        params["p"] = random.choice([0.5, 1.0/3.0, 2.0, 3.0])
    elif transformer_name == "groupby_quantile":
        params["q"] = random.choice([0.25, 0.75])
        
    return Gene(transformer_name=transformer_name, input_cols=input_cols, params=params)

@dataclass
class Gene:
    """
    Represents a single feature engineering step.
    """
    transformer_name: str
    input_cols: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    state: Optional[Any] = None
    tested: bool = False

    def __post_init__(self):
        transformer = evo_transformers.get(self.transformer_name)
        if transformer is not None:
            transformer.params = self.params
    
    @property
    def output_col(self) -> str:
        transformer = evo_transformers.get(self.transformer_name)
        if transformer is None:
            raise ValueError(f"Transformer {self.transformer_name} not found in registry.")
        try:
            return transformer.name_generator(self.input_cols, params=self.params)
        except TypeError:
            return transformer.name_generator(self.input_cols)

    def to_formula(self) -> str:
        return gene_to_formula(self)

def gene_to_formula(gene: Gene, truncate: bool = True) -> str:
    cols = gene.input_cols
    if truncate and len(cols) > 3:
        cols_str = ", ".join(cols[:3]) + f", ... + {len(cols) - 3} more"
    else:
        cols_str = ", ".join(cols)
    
    params = gene.params
    transformer_name = gene.transformer_name
    
    if transformer_name == "one_hot_encode":
        comp_idx = params.get("comp_idx")
        comp_str = "other" if comp_idx == 6 else str(comp_idx)
        return f"ohe_{comp_str}({cols_str})"
    elif params.get("component") is not None:
        return f"{transformer_name}_{params.get('component')}({cols_str})"
    elif params.get("comp_idx") is not None:
        if transformer_name == "genie_centroid_dist":
            gini_threshold = params.get("gini_threshold")
            k_val = params.get("k")
            comp_idx = params.get("comp_idx")
            return f"genie_cdist{comp_idx}_k{k_val}_t{gini_threshold:.2f}({cols_str})"
        elif transformer_name == "lumbermark_centroid_dist":
            comp_idx = params.get("comp_idx")
            k_val = params.get("k")
            return f"lumb_cdist{comp_idx}_k{k_val}({cols_str})"
        else:
            comp_idx = params.get("comp_idx")
            return f"{transformer_name}{comp_idx}({cols_str})"
    elif params.get("Q") is not None:
        return f"{transformer_name}{params.get('Q')}({cols_str})"
    elif params.get("base") is not None:
        return f"{transformer_name}{params.get('base')}({cols_str})"
    elif params.get("p") is not None:
        p_val = params.get("p")
        return f"pow{p_val:.4g}({cols_str})"
    elif params.get("q") is not None:
        return f"{transformer_name}_q{params.get('q'):.2f}({cols_str})"
    elif params.get("k") is not None:
        if transformer_name == "genie" and params.get("gini_threshold") is not None:
            return f"genie_k{params.get('k')}_t{params.get('gini_threshold'):.2f}({cols_str})"
        else:
            return f"{transformer_name}_k{params.get('k')}({cols_str})"
    else:
        return f"{transformer_name}({cols_str})"

def gene_to_state_formula(gene: Gene) -> str:
    if gene.transformer_name in ["pca", "truncated_svd", "umap", "genie_centroid_dist", "lumbermark_centroid_dist"]:
        return f"{gene.transformer_name}({', '.join(gene.input_cols)})"
    else:
        return gene_to_formula(gene, truncate=False)

def individual_to_recipe_string(individual: 'Individual') -> str:
    if not individual.genes:
        return "[Original features only]"
    formulas = [gene_to_formula(g) for g in individual.genes]
    return "[" + ", ".join(formulas) + "]"

def crossover(ind1: 'Individual', ind2: 'Individual', verbose: bool = False) -> 'Individual':
    genes1 = [g for g in ind1.genes if random.random() < 0.5]
    genes2 = [g for g in ind2.genes if random.random() < 0.5]
    child_genes = genes1 + genes2
    
    seen_outputs = set()
    deduped_genes = []
    for g in child_genes:
        out_col = g.output_col
        if out_col not in seen_outputs:
            seen_outputs.add(out_col)
            import copy
            ng = copy.deepcopy(g)
            ng.state = None
            ng.tested = False
            deduped_genes.append(ng)
            
    child = Individual(
        numeric_cols=list(ind1.numeric_cols),
        categorical_cols=list(ind1.categorical_cols),
        genes=deduped_genes
    )
    child._topological_sort()
    return child

def union_crossover(ind1: 'Individual', ind2: 'Individual', verbose: bool = False) -> 'Individual':
    child_genes = list(ind1.genes) + list(ind2.genes)
    
    seen_outputs = set()
    deduped_genes = []
    for g in child_genes:
        out_col = g.output_col
        if out_col not in seen_outputs:
            seen_outputs.add(out_col)
            import copy
            ng = copy.deepcopy(g)
            ng.state = None
            ng.tested = False
            deduped_genes.append(ng)
            
    child = Individual(
        numeric_cols=list(ind1.numeric_cols),
        categorical_cols=list(ind1.categorical_cols),
        genes=deduped_genes
    )
    child._topological_sort()
    return child


class Individual:
    """
    Represents a candidate feature engineering pipeline.
    """
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str], genes: Optional[List[Gene]] = None):
        self.id = str(uuid.uuid4())
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.genes = genes if genes is not None else []
        self.fitness: float = float('nan')

    def _topological_sort(self):
        """
        Sort genes so that dependencies (output cols of previous genes) are evaluated in order.
        """
        available = set(self.numeric_cols + self.categorical_cols)
        sorted_genes = []
        unsorted = list(self.genes)
        
        while unsorted:
            progress = False
            for i, gene in enumerate(unsorted):
                if all(c in available for c in gene.input_cols):
                    sorted_genes.append(gene)
                    available.add(gene.output_col)
                    unsorted.pop(i)
                    progress = True
                    break
            if not progress:
                # Cyclic dependency or missing column, drop the invalid genes
                break
                
        self.genes = sorted_genes

    def crossover(self, other: 'Individual', verbose: bool = False) -> 'Individual':
        return crossover(self, other, verbose)

    def union_crossover(self, other: 'Individual', verbose: bool = False) -> 'Individual':
        return union_crossover(self, other, verbose)

    def mutate(self, verbose=False, force_add=False, importances=None, temperature=1.0, allowed_transformers=None):
        """
        Mutates the individual in-place by adding, modifying, or removing a gene using feature importances.
        """
        if not self.numeric_cols and not self.categorical_cols:
            return
            
        if importances is None:
            importances = {}
            
        def weighted_sample(cols, size, replace=False):
            if not cols:
                return []
            if len(cols) == 1:
                return [cols[0]] * size
            if not importances:
                return random.choices(cols, k=size) if replace else random.sample(cols, min(size, len(cols)))
                
            # Baseline for missing/new features
            known_vals = [v for k, v in importances.items() if k in cols and v > 0]
            baseline = min(known_vals) if known_vals else 0.01
            
            import math
            weights = []
            for c in cols:
                val = importances.get(c, baseline)
                if val <= 0: val = baseline
                weights.append(math.exp(val / temperature))
                
            if sum(weights) == 0:
                weights = [1.0] * len(cols)
                
            if replace:
                return random.choices(cols, weights=weights, k=size)
            else:
                # Weighted sample without replacement
                res = []
                pool = list(cols)
                pool_weights = list(weights)
                for _ in range(min(size, len(cols))):
                    c = random.choices(pool, weights=pool_weights, k=1)[0]
                    res.append(c)
                    idx = pool.index(c)
                    pool.pop(idx)
                    pool_weights.pop(idx)
                    if sum(pool_weights) == 0:
                        pool_weights = [1.0] * len(pool)
                return res

        mut_type = 3 # Default Add
        if not force_add and len(self.genes) > 0:
            r = random.random()
            if r < 0.33:
                mut_type = 1 # Remove
            elif r < 0.66:
                mut_type = 2 # Modify
                
        if mut_type == 1:
            # Importance-guided Removal
            if importances and len(self.genes) > 1:
                gene_imps = [importances.get(g.output_col, 0) for g in self.genes]
                # Inverse importance: low importance = high probability of removal
                weights = [1.0 / (imp + 1e-8) for imp in gene_imps]
                idx = random.choices(range(len(self.genes)), weights=weights, k=1)[0]
            else:
                idx = random.randint(0, len(self.genes) - 1)
                
            if verbose:
                print(f"Removed gene: {self.genes[idx].to_formula()}")
            self.genes.pop(idx)
            self.fitness = float('nan')
            
        elif mut_type == 2:
            # Enriched Modify mutation: pick a random gene, then swap a col or mutate its params
            idx = random.randint(0, len(self.genes) - 1)
            gene = self.genes[idx]
            transformer = evo_transformers[gene.transformer_name]
            
            applicable_ops = []
            
            avail_cols = list(self.numeric_cols) if transformer.input_type == "numeric" else (
                list(self.categorical_cols) if transformer.input_type == "categorical"
                else list(self.numeric_cols) + list(self.categorical_cols)
            )
            # Only tested genes may contribute derived columns (R invariant)
            for prev_g in self.genes[:idx]:
                if prev_g.tested:
                    avail_cols.append(prev_g.output_col)
            
            pool = list(avail_cols)
            if not getattr(transformer, 'allow_replace', False):
                pool = [c for c in pool if c not in gene.input_cols]
                
            if gene.input_cols and pool:
                applicable_ops.append("swap_col")
                
            if gene.params:
                applicable_ops.append("mutate_params")
                
            sub_op = random.choice(applicable_ops) if applicable_ops else None

            
            if sub_op == "swap_col":
                new_col = weighted_sample(pool, 1, replace=False)[0]
                idx_to_replace = random.randint(0, len(gene.input_cols) - 1)
                gene.input_cols[idx_to_replace] = new_col
                gene.state = None    # must re-fit after input change
                gene.tested = False  # R invariant: reset tested on any gene change
                
            elif sub_op == "mutate_params":
                param_name = random.choice(list(gene.params.keys()))
                old_val = gene.params[param_name]
                
                if param_name == "k":
                    candidates = [c for c in range(2, 6) if c != old_val]
                    gene.params["k"] = random.choice(candidates) if candidates else old_val
                elif param_name == "gini_threshold":
                    gene.params["gini_threshold"] = round(random.uniform(0.1, 0.9), 2)
                elif param_name == "Q":
                    candidates = [c for c in range(3, 11) if c != old_val]
                    gene.params["Q"] = random.choice(candidates) if candidates else old_val
                elif param_name == "base":
                    candidates = [c for c in range(2, 11) if c != old_val]
                    gene.params["base"] = random.choice(candidates) if candidates else old_val
                elif param_name == "component":
                    components = ["year", "month", "day", "hour", "day_of_week", "weekend"]
                    candidates = [c for c in components if c != old_val]
                    gene.params["component"] = random.choice(candidates) if candidates else old_val
                elif param_name == "p":
                    candidates = [c for c in [0.5, 1.0/3.0, 2.0, 3.0] if c != old_val]
                    gene.params["p"] = random.choice(candidates) if candidates else old_val
                elif param_name == "q":
                    candidates = [c for c in [0.25, 0.75] if c != old_val]
                    gene.params["q"] = random.choice(candidates) if candidates else old_val
                
                gene.state = None    # must re-fit after param change
                gene.tested = False  # R invariant: reset tested on any gene change
                
            self.fitness = float('nan')
            
        else:
            # Add: select features intelligently
            if allowed_transformers:
                t_name = random.choice(allowed_transformers)
            else:
                t_name = random.choice(list(evo_transformers.keys()))
                
            transformer = evo_transformers[t_name]
            
            avail_cols = list(self.numeric_cols) if transformer.input_type == "numeric" else (
                list(self.categorical_cols) if transformer.input_type == "categorical" else self.numeric_cols + self.categorical_cols
            )
            # Add outputs from already-tested genes only (R invariant)
            for g in self.genes:
                if g.tested:
                    avail_cols.append(g.output_col)
            
            if avail_cols:
                if transformer.type_ in ["binary", "mixed_binary"]:
                    if len(avail_cols) < 2:
                        return  # can't build a binary transformer with < 2 cols
                    n_inputs = 2
                elif transformer.type_ == "multivariate":
                    n_inputs = random.randint(2, max(2, min(5, len(avail_cols))))
                else:
                    n_inputs = 1
                
                selected_cols = weighted_sample(avail_cols, n_inputs, replace=transformer.allow_replace)
                
                new_gene = create_gene(t_name, selected_cols)
                self.genes.append(new_gene)
                self.fitness = float('nan')
                
        self._topological_sort()
