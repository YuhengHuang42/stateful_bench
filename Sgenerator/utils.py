from functools import reduce
from operator import getitem
import json
import ast
from typing import Dict, List, Tuple, Set, Any, Optional
import inspect
import sys
from stdlib_list import stdlib_list
import importlib
import builtins
import math
import networkx as nx

def load_file(file_path: str):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))
    return result

def get_nested_value(dct: dict, key_list: list):
    """Safely get nested value from dictionary using list of keys"""
    try:
        return reduce(getitem, key_list, dct)
    except (KeyError, TypeError):
        return None  # Or raise appropriate exception

def get_nested_path_string(dct_str: str, key_list: list):
    """Return string representation of nested dictionary access path"""
    path = dct_str
    for key in key_list:
        if isinstance(key, str):
            path += f'["{key}"]'
        else:
            path += f'[{key}]'
    return path

def get_added_changes(old_book, new_book):
    """Compute diff between two occurrence books (works with both dicts and OccurenceBook)."""
    old_get = old_book.get if hasattr(old_book, 'get') else lambda k, d=0: old_book.get(k, d)
    changes = {}
    for k, v in new_book.items():
        old_val = old_get(k, 0)
        if old_val == 0:
            changes[k] = v
        else:
            assert v >= old_val, f"The number of occurence of {k} is decreased from {old_val} to {v}."
            if v > old_val:
                changes[k] = v - old_val
    return changes

def write_jsonl(output_path, data):
    with open(output_path, "w") as ifile:
        for entry in data:
            json_line = json.dumps(entry)
            ifile.write(json_line + '\n')

def merge_dicts(*dicts):
    """
    Merge multiple dictionaries into a single dictionary.
    If there are duplicate keys, the value from the later dictionary will override the earlier one.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        dict: Merged dictionary
    """
    result = {}
    for d in dicts:
        if d is not None:
            result.update(d)
    return result

# ===== OPENAI API Related Functions=====
def generate_jsonl_for_openai(request_id_list, 
                              message_list, 
                              output_path=None,
                              max_tokens=None, 
                              model_type="gpt-4.1", 
                              url="/v1/chat/completions"):
    """
    Prepare input batch data for OPENAI API
    Args:
        request_id_list: used to index the message_list.
        message_list: list of messages to be sent to the API
        output_path: output file path
        max_tokens: maximum tokens to generate
        model_type: model type of OPENAI API
        url: API endpoint
    """
    assert len(request_id_list) == len(message_list)
    data = []
    requestid_to_message = dict(zip(request_id_list, message_list))
    for idx, item in enumerate(request_id_list):
        request_id = item
        message = message_list[idx]
        body = {"model": model_type, 
         "messages": message
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        per_request = {
            "custom_id": request_id,
            "method": "POST",
            "url": url,
            "body": body
        }
        data.append(per_request)
        
    if output_path is not None:
        write_jsonl(output_path, data)
    
    return data, requestid_to_message

def submit_batch_request_openai(client, 
                                input_file_path, 
                                url="/v1/chat/completions", 
                                completion_window="24h", 
                                description="code analysis"):
    """
    Submit the batch task to OPENAI API
    Args:
        client: OPENAI API client (client = openai.OpenAI(api_key=api_key))
        input_file_path: input file path
        url: API endpoint
        completion_window: completion window
        description: description of the submitted task
    """
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_submit_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=url,
        completion_window=completion_window,
        metadata={
        "description": description
        }
    )
    batch_submit_info_id = batch_submit_info.id
    batch_result_info = client.batches.retrieve(batch_submit_info_id)
    
    return (batch_submit_info, batch_result_info)

# Define built-in functions and modules to exclude from analysis
# This includes functions from the builtins module and common libraries
# TODO: Update this list as needed to include any additional libraries ------------------------
BUILTIN_FUNCS = set(dir(builtins))
#EXCLUDE_PREFIXES = {"math", "json", "builtins"}
EXCLUDE_FUNC_NAMES = set(dir(math))
stdlib_modules = stdlib_list("3.9")

def is_stdlib_function(func, library_list=stdlib_modules):
    module = inspect.getmodule(func)
    if module is None:
        return False  # Built-in or dynamically defined

    module_name = module.__name__.split('.')[0]
    return module_name in library_list


def is_stdlib_from_string(name: str) -> bool:
    parts = name.split(".")
    try:
        # Step 1: Import the top-level module
        module = importlib.import_module(parts[0])
        
        # Step 2: Traverse attributes (e.g., math.sqrt)
        obj = module
        for part in parts[1:]:
            obj = getattr(obj, part)
        
        # Step 3: Get the module the object is defined in
        obj_module = inspect.getmodule(obj)
        if obj_module is None:
            return True  # Possibly built-in like `len`

        modname = obj_module.__name__.split('.')[0]
        return modname in stdlib_modules

    except (ImportError, AttributeError):
        return False
    
class APICallAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.var_def: Dict[str, Set[Tuple[str, int]]] = {}  # var_name -> set of (api_func, api_lineno) origins (transitive)
        self.var_direct_def: Dict[str, Set[Tuple[str, int]]] = {}  # var_name -> set of (api_func, api_lineno) that directly assigned it (non-transitive)
        self.calls_set: Set[Tuple[Optional[str], Tuple[str, ...], str, int]] = set() # (assigned_var, sorted_args_tuple, func_name, lineno)
        self.control_stack_depth = 0
        self.control_gated_calls: Set[Tuple[str, int]] = set() # (func_name, lineno)
        self.calls_in_order: List[Tuple[Optional[str], Tuple[str, ...], str, int]] = []
        self.call_receiver: Dict[Tuple[str, int], Optional[str]] = {}  # (func_name, lineno) -> receiver variable
        # Per-call snapshot of argument origins, captured at visit time so that
        # if/else branch overwrites don't corrupt the data.
        # Key: (func_name, lineno)  Value: Dict[arg_var -> set of (api_func, api_lineno)]
        self.call_arg_direct_origins: Dict[Tuple[str, int], Dict[str, Set[Tuple[str, int]]]] = {}
    
    '''
    def _is_excluded(self, func_name: str) -> bool:
        # More robust exclusion, e.g., common builtins or non-API methods
        excluded_prefixes = ('__',)
        excluded_exact = {'print', 'len', 'range', 'sorted', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'open'}
        
        if func_name in excluded_exact:
            return True
        if any(func_name.startswith(p) for p in excluded_prefixes):
            return True
        # Example: if 'json' is a method, not a top-level API call, exclude it if it's not from a known API module
        if '.' in func_name and func_name.endswith('.json'): # Heuristic, might need refinement
             # If 'requests.Response.json' is an API, this needs to be smarter
             # For now, let's assume simple method calls like .json() on generic objects are not primary APIs
             # unless qualified like 'requests.Response.json'
            if not func_name.startswith('requests.'): # crude check
                return True
        return False
    '''
    
    def _record_call(self, assigned_var: Optional[str], args_tuple: Tuple[str, ...],
                     func_name: str, lineno: int, call_node: ast.Call):
        """Centralized call registration that keeps all bookkeeping in sync."""
        entry = (assigned_var, args_tuple, func_name, lineno)
        if entry not in self.calls_set:
            self.calls_set.add(entry)
            self.calls_in_order.append(entry)
        receiver = self._get_receiver_var(call_node.func)
        self.call_receiver[(func_name, lineno)] = receiver
        if self.control_stack_depth > 0:
            self.control_gated_calls.add((func_name, lineno))
        # Snapshot the current direct origins for each argument variable so
        # that later if/else branch overwrites don't corrupt the data.
        key = (func_name, lineno)
        if key not in self.call_arg_direct_origins:
            arg_origins: Dict[str, Set[Tuple[str, int]]] = {}
            for arg_var in args_tuple:
                if arg_var in self.var_direct_def:
                    arg_origins[arg_var] = set(self.var_direct_def[arg_var])
                elif arg_var in self.var_def:
                    arg_origins[arg_var] = set()  # defined but no API origin
                # else: not in scope — handled in extract_state_transitions
            self.call_arg_direct_origins[key] = arg_origins

    def _is_excluded(self, func_name: str) -> bool:
        if func_name in BUILTIN_FUNCS:
            return True
        if is_stdlib_from_string(func_name):
            return True
        if func_name in EXCLUDE_FUNC_NAMES:
            return True
        return False

    def _get_qualified_name(self, node: ast.expr) -> str:
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        elif isinstance(current, ast.Call): # e.g. func_call().attr
            pass
        return '.'.join(reversed(parts)) if parts else '<unknown>'

    def _get_receiver_var(self, node: ast.expr) -> Optional[str]:
        """Extract the receiver variable of a method call.

        For ``obj.method()``, returns ``"obj"``.
        For ``obj.sub.method()``, returns ``"obj"``.
        For plain ``func()``, returns ``None``.
        """
        if not isinstance(node, ast.Attribute):
            return None
        base = node.value
        while isinstance(base, ast.Attribute):
            base = base.value
        if isinstance(base, ast.Name):
            return base.id
        return None

    def _collect_variable_refs(self, node: Optional[ast.AST]) -> List[str]:
        refs: List[str] = []
        if node is None:
            return refs
        if isinstance(node, ast.Name):
            refs.append(node.id)
        # Broader check for common AST nodes that can contain Name nodes
        elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Subscript, 
                               ast.Call, ast.Attribute, ast.List, ast.Tuple, 
                               ast.Dict, ast.Starred, ast.FormattedValue, ast.JoinedStr,
                               ast.Lambda)):
            for child in ast.iter_child_nodes(node):
                refs.extend(self._collect_variable_refs(child))
        elif isinstance(node, ast.keyword): # For keyword arguments' values
             refs.extend(self._collect_variable_refs(node.value))
        return list(set(refs)) # Return unique refs

    def _get_arguments(self, node: ast.Call) -> List[str]:
        args_refs: List[str] = []
        for arg_node in node.args:
            args_refs.extend(self._collect_variable_refs(arg_node))
        for kw_node in node.keywords:
            args_refs.extend(self._collect_variable_refs(kw_node.value)) # kw.arg is the name, kw.value is the expression
        # Also consider variable references in the function itself if it's complex, e.g., obj.method where obj is a var
        args_refs.extend(self._collect_variable_refs(node.func))
        return list(set(args_refs)) # Return unique refs

    def visit_Assign(self, node: ast.Assign):
        target_var_names: List[str] = []
        if isinstance(node.targets[0], ast.Tuple):
            for target_element_node in node.targets[0].elts:
                if isinstance(target_element_node, ast.Name):
                    target_var_names.append(target_element_node.id)
        else:
            for target_node in node.targets:
                if isinstance(target_node, ast.Name):
                    target_var_names.append(target_node.id)

        current_origins: Set[Tuple[str, int]] = set()
        direct_origins: Set[Tuple[str, int]] = set()

        primary_assigned_var = target_var_names[0] if target_var_names else None

        for sub_node in ast.walk(node.value):
            if isinstance(sub_node, ast.Call):
                func_name = self._get_qualified_name(sub_node.func)
                if not self._is_excluded(func_name):
                    current_origins.add((func_name, sub_node.lineno))
                    direct_origins.add((func_name, sub_node.lineno))
                    
                    args = self._get_arguments(sub_node)
                    args_tuple = tuple(sorted(list(set(args))))
                    
                    assigned_to_this_specific_call = None
                    if sub_node == node.value:
                         assigned_to_this_specific_call = primary_assigned_var
                    
                    self._record_call(assigned_to_this_specific_call, args_tuple,
                                      func_name, sub_node.lineno, sub_node)
        
        referenced_vars_in_rhs = self._collect_variable_refs(node.value)
        for ref_var in referenced_vars_in_rhs:
            current_origins.update(self.var_def.get(ref_var, set()))

        for var_name in target_var_names:
            self.var_def[var_name] = current_origins.copy()
            if len(direct_origins) > 0:
                self.var_direct_def[var_name] = direct_origins.copy()
            else:
                # No direct API call on RHS — propagate direct defs from
                # referenced variables (e.g. y = x where x was from an API).
                inherited: Set[Tuple[str, int]] = set()
                for ref_var in referenced_vars_in_rhs:
                    inherited.update(self.var_direct_def.get(ref_var, set()))
                self.var_direct_def[var_name] = inherited
            
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            func_name = self._get_qualified_name(node.value.func)
            if not self._is_excluded(func_name):
                args = self._get_arguments(node.value)
                args_tuple = tuple(sorted(list(set(args))))
                self._record_call(None, args_tuple, func_name, node.value.lineno, node.value)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        iterable_origins: Set[Tuple[str, int]] = set()
        iterable_direct_origins: Set[Tuple[str, int]] = set()
        iterable_referenced_vars = self._collect_variable_refs(node.iter)
        for ref_var in iterable_referenced_vars:
            iterable_origins.update(self.var_def.get(ref_var, set()))
            iterable_direct_origins.update(self.var_direct_def.get(ref_var, set()))
        
        if isinstance(node.iter, ast.Call):
            func_name = self._get_qualified_name(node.iter.func)
            if not self._is_excluded(func_name):
                iterable_origins.add((func_name, node.iter.lineno))
                iterable_direct_origins.add((func_name, node.iter.lineno))
                args = self._get_arguments(node.iter)
                args_tuple = tuple(sorted(list(set(args))))
                self._record_call(None, args_tuple, func_name, node.iter.lineno, node.iter)


        loop_vars: List[str] = []
        if isinstance(node.target, ast.Name):
            loop_vars.append(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    loop_vars.append(elt.id)
        
        for loop_var in loop_vars:
            self.var_def[loop_var] = self.var_def.get(loop_var, set()).union(iterable_origins)
            self.var_direct_def[loop_var] = self.var_direct_def.get(loop_var, set()).union(iterable_direct_origins)

        self.control_stack_depth += 1
        # Visit orelse first if it exists, as loop var is not in scope there with new val
        if node.orelse:
            for child_node in node.orelse:
                self.visit(child_node)
        # Visit body
        for child_node in node.body:
            self.visit(child_node) # Use self.visit to ensure all node types are handled by visitors
        self.control_stack_depth -= 1

    def visit_If(self, node: ast.If):
        self.control_stack_depth += 1
        self.generic_visit(node)
        self.control_stack_depth -= 1

    def visit_While(self, node: ast.While):
        self.control_stack_depth += 1
        self.generic_visit(node)
        self.control_stack_depth -= 1

    def visit_Try(self, node: ast.Try):
        self.control_stack_depth += 1
        self.generic_visit(node)
        self.control_stack_depth -= 1
        
    def visit_With(self, node: ast.With):
        self.control_stack_depth += 1
        # Process with items for assignments (e.g., with open(...) as f:)
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                context_origins: Set[Tuple[str, int]] = set()
                context_direct_origins: Set[Tuple[str, int]] = set()
                
                for sub_node in ast.walk(item.context_expr):
                    if isinstance(sub_node, ast.Call):
                        func_name = self._get_qualified_name(sub_node.func)
                        if not self._is_excluded(func_name):
                            context_origins.add((func_name, sub_node.lineno))
                            context_direct_origins.add((func_name, sub_node.lineno))
                            args = self._get_arguments(sub_node)
                            args_tuple = tuple(sorted(list(set(args))))
                            assigned_to = item.optional_vars.id if isinstance(item.context_expr, ast.Call) and self._get_qualified_name(item.context_expr.func) == func_name and item.context_expr.lineno == sub_node.lineno else None
                            self._record_call(assigned_to, args_tuple, func_name,
                                              sub_node.lineno, sub_node)
                
                ref_vars = self._collect_variable_refs(item.context_expr)
                for ref_var in ref_vars:
                    context_origins.update(self.var_def.get(ref_var, set()))
                
                self.var_def[item.optional_vars.id] = context_origins
                if len(context_direct_origins) > 0:
                    self.var_direct_def[item.optional_vars.id] = context_direct_origins
                else:
                    inherited: Set[Tuple[str, int]] = set()
                    for ref_var in ref_vars:
                        inherited.update(self.var_direct_def.get(ref_var, set()))
                    self.var_direct_def[item.optional_vars.id] = inherited
                
        self.generic_visit(node) # Visit body
        self.control_stack_depth -= 1

    def _get_module_prefix(self, func_name: str) -> str:
        if not func_name or func_name == '<unknown>':
            return "<unknown_module>"
        # Improved to handle common patterns like 'module.submodule.function'
        parts = func_name.split('.')
        # Heuristic: if it's like 'ClassName.method', the ClassName might be the "module" in some contexts,
        # but for APIs, we usually care about the imported module like 'requests' or 'sklearn.linear_model'.
        # This simple split takes the first part.
        if len(parts) > 1:
            # Consider 'sklearn.linear_model.LinearRegression' -> 'sklearn' or 'sklearn.linear_model'
            # For now, let's take 'requests' from 'requests.get' or 'sklearn' from 'sklearn.svm.SVC'
            if parts[0].isidentifier(): # check if it's a valid module name start
                 return parts[0]
        return "<root_or_builtin>" # If no clear module prefix (e.g. 'open()', or 'method' if not qualified)


    def compute_metrics(self) -> Dict[str, Any]:
        # Convert calls_set to a list of dictionaries for easier processing if needed, or use as is.
        # For this computation, the tuple structure is fine.
        # self.calls_set elements: (assigned_var, sorted_args_tuple, func_name, lineno)
        
        actual_calls = list(self.calls_set)

        binding_edges_set: Set[Tuple[str, Optional[str]]] = set()
        cross_module_edges_set: Set[Tuple[str, str]] = set()

        for assigned_var_call_B, args_call_B_tuple, func_B, lineno_B in actual_calls:
            module_B = self._get_module_prefix(func_B)
            args_call_B_list = list(args_call_B_tuple)

            for arg_var_in_B in args_call_B_list:
                origins_of_arg = self.var_def.get(arg_var_in_B, set())
                if origins_of_arg: # If arg_var_in_B has any API origin
                    # This means arg_var_in_B (which came from some API call A) is used in API call B
                    # which defines assigned_var_call_B.
                    # The binding edge is (arg_var_in_B, assigned_var_call_B)
                    if assigned_var_call_B is not None: # Only form edge if B assigns to a var
                        binding_edges_set.add((arg_var_in_B, assigned_var_call_B))

                    for func_A, lineno_A in origins_of_arg:
                        module_A = self._get_module_prefix(func_A)
                        if module_A and module_B and module_A != module_B and module_A != "<unknown_module>" and module_B != "<unknown_module>":
                            cross_module_edges_set.add(tuple(sorted((module_A, module_B))))


        binding_count = len(binding_edges_set)
        
        # Calculate control_gated_ratio
        # total_unique_api_calls_invoked: set of (func_name, lineno)
        total_unique_api_calls_invoked = {(call_info[2], call_info[3]) for call_info in actual_calls}
        
        # control_gated_calls already stores (func_name, lineno) of calls within control blocks
        gated_api_calls = total_unique_api_calls_invoked & self.control_gated_calls
        control_gated_ratio = len(gated_api_calls) / max(1, len(total_unique_api_calls_invoked)) if total_unique_api_calls_invoked else 0

        return {
            "binding_count": binding_count,
            "binding_edges": list(binding_edges_set),
            "cross_module_edges": list(cross_module_edges_set), # Added this metric
            "cross_component_ratio": len(cross_module_edges_set) / max(1, binding_count) if binding_count > 0 else 0, # Ratio
            "control_gated_ratio": control_gated_ratio,
            "control_gated_calls": list(self.control_gated_calls), # All (func,lineno) in control blocks
            "total_api_calls_invoked": len(total_unique_api_calls_invoked), # Count of unique (func,lineno) API calls
            "all_calls_details": actual_calls, # For debugging
            "var_definitions_origins": self.var_def # For debugging
        }

    def extract_state_transitions(self) -> Dict[Tuple[str, str], int]:
        """Identify (API_a, API_b) state transitions from the analysed code.

        A transition ``(API_a, API_b)`` is emitted when the two API calls are
        *bound* by shared state — i.e. they are not independent.  Two binding
        criteria are checked:

        1. **Producer-consumer**: ``API_a`` produces a value (assigned to some
           variable) that later flows into ``API_b`` as an argument.  The
           existing ``var_def`` taint map already tracks which variables
           originate from which API calls, so when ``API_b`` consumes a
           variable whose origin set contains ``API_a``, the pair is recorded.

        2. **Same-object interaction**: ``API_a`` and ``API_b`` are method
           calls on the same receiver variable (e.g. ``obj.read()`` followed
           by ``obj.write()``).  This captures the common pattern where two
           APIs mutate / query the same local or remote object without an
           explicit producer-consumer data-flow edge.

        Returns
        -------
        Dict[Tuple[str, str], int]
            Mapping from ``(API_a_name, API_b_name)`` to the number of times
            that transition was observed.  The format is identical to
            ``OccurenceBook.get_round_transitions()``.

            When ``API_b`` consumes a variable that was defined in the program
            but has no API origin (e.g. initialised from a literal or an
            excluded call like ``torch.randn``), the pair ``("NONE", API_b)``
            is emitted — mirroring the ``"NONE"`` sentinel used by
            ``Schema.form_pair_transition`` for state variables with no prior
            transitions.
        """
        transitions: Dict[Tuple[str, str], int] = {}

        def _inc(pair: Tuple[str, str]):
            transitions[pair] = transitions.get(pair, 0) + 1

        calls = self.calls_in_order
        # calls elements: (assigned_var, sorted_args_tuple, func_name, lineno)

        # Build the set of all API-call-produced variable names for NONE detection.
        api_produced_vars: Set[str] = set()
        for assigned_var, _args, _func, _lineno in calls:
            if assigned_var is not None:
                api_produced_vars.add(assigned_var)

        # Modules / imports that should not be treated as state variables.
        _KNOWN_MODULES = frozenset(('torch', 'np', 'json', 'requests', 'os', 'sys', 'math'))

        # --- 1. Producer-consumer edges via call_arg_direct_origins ---
        # Uses the per-call snapshot captured at visit time so that if/else
        # branch variable overwrites do not corrupt the data.
        for _assigned_var_B, args_B, func_B, lineno_B in calls:
            snapshot = self.call_arg_direct_origins.get((func_B, lineno_B), {})
            for arg_var in args_B:
                if arg_var in _KNOWN_MODULES:
                    continue
                if arg_var in snapshot:
                    direct_origins = snapshot[arg_var]
                    if len(direct_origins) > 0:
                        for func_A, lineno_A in direct_origins:
                            if (func_A, lineno_A) != (func_B, lineno_B):
                                _inc((func_A, func_B))
                    else:
                        # Variable is defined in the program but has no API
                        # origin — it came from initialisation.
                        _inc(("NONE", func_B))
                else:
                    # Variable not in the snapshot — either not assigned in
                    # the analysed code (pre-loaded / externally injected) or
                    # assigned after this call.  Treat user-looking variables
                    # as NONE-origin.
                    if (arg_var not in api_produced_vars
                            and not arg_var[0].isupper()):
                        _inc(("NONE", func_B))

        # --- 2. Same-object (receiver) edges ---
        # Group calls by receiver variable; within each group, consecutive
        # pairs in source order form transitions.  Skip module-level
        # receivers (e.g. ``torch``) since calling two functions on a
        # module does not imply shared mutable state.
        receiver_groups: Dict[str, List[Tuple[str, int]]] = {}
        for _assigned_var, _args, func_name, lineno in calls:
            recv = self.call_receiver.get((func_name, lineno))
            if recv is not None and recv not in _KNOWN_MODULES:
                receiver_groups.setdefault(recv, []).append((func_name, lineno))

        for recv_var, call_list in receiver_groups.items():
            for i in range(len(call_list) - 1):
                func_A = call_list[i][0]
                func_B = call_list[i + 1][0]
                if func_A != func_B or call_list[i][1] != call_list[i + 1][1]:
                    _inc((func_A, func_B))

        return transitions


def extract_state_transitions(code: str) -> Dict[Tuple[str, str], int]:
    """Static analysis entry point: extract (API_a, API_b) state transitions.

    Given a piece of Python source code, this function parses it, performs
    taint-style data-flow analysis to identify API calls that are bound by
    shared state, and returns the set of transition pairs with counts.

    A transition ``(API_a, API_b)`` is created when:

    1. **Producer-consumer** — ``API_a`` is called first and its output flows
       (directly or transitively through variables) into ``API_b``.
    2. **Same-object interaction** — ``API_a`` and ``API_b`` are both method
       calls on the same receiver variable (local or remote object).

    Parameters
    ----------
    code : str
        Python source code to analyse.

    Returns
    -------
    Dict[Tuple[str, str], int]
        Mapping from ``(API_a_name, API_b_name)`` to the number of times the
        transition was observed.  The structure mirrors
        ``OccurenceBook.get_round_transitions()``.  When an API call operates
        on a variable with no prior API origin (initialised state), the pair
        ``("NONE", API_b_name)`` is emitted.

    Examples
    --------
    >>> transitions = extract_state_transitions('''
    ... import requests
    ... resp = requests.get("http://example.com/api/sessions")
    ... data = resp.json()
    ... requests.post("http://example.com/api/update", json=data)
    ... ''')
    >>> ("requests.get", "resp.json") in transitions
    True
    >>> ("resp.json", "requests.post") in transitions
    True
    """
    tree = ast.parse(code)
    analyzer = APICallAnalyzer()
    analyzer.visit(tree)
    return analyzer.extract_state_transitions()


def extract_transition_chains(code: str,
                              include_none: bool = True,
                              ) -> List[List[str]]:
    """Extract all maximal transition chains from a piece of Python code.

    A chain is a sequence ``[API_a, API_b, ..., API_z]`` where each
    consecutive pair ``(API_i, API_{i+1})`` is a state transition identified
    by ``extract_state_transitions``.  Chains are found by building a directed
    graph from the pairwise transitions and enumerating all maximal simple
    paths (paths that cannot be extended at either end).

    Parameters
    ----------
    code : str
        Python source code to analyse.
    include_none : bool, default True
        If *True*, the ``"NONE"`` sentinel is kept as a chain root so chains
        look like ``["NONE", "torch.permute", "torch.cat"]``.  If *False*,
        ``"NONE"`` nodes are removed and chains start from the first real API.

    Returns
    -------
    List[List[str]]
        Each inner list is one maximal chain of API names.  Chains are sorted
        longest-first, then lexicographically.

    Examples
    --------
    >>> chains = extract_transition_chains('''
    ... import torch
    ... x = torch.randn(3, 4)
    ... y = torch.permute(x, (1, 0))
    ... z = torch.transpose(y, 0, 1)
    ... ''')
    >>> any("torch.permute" in c and "torch.transpose" in c for c in chains)
    True
    """
    tree = ast.parse(code)
    analyzer = APICallAnalyzer()
    analyzer.visit(tree)
    transitions = analyzer.extract_state_transitions()
    return _chains_from_transitions(transitions, include_none=include_none)


def _chains_from_transitions(transitions: Dict[Tuple[str, str], int],
                             include_none: bool = True,
                             ) -> List[List[str]]:
    """Build maximal chains from a pairwise transition dict.

    Shared by both the top-level helper and any caller that already has
    the transitions dict (e.g. from ``OccurenceBook.get_round_transitions``).
    """
    G = nx.DiGraph()
    for (a, b) in transitions:
        G.add_edge(a, b)

    if not G:
        return []

    # Source nodes: nodes with no incoming edge (or only self-loops).
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    # Sink nodes: nodes with no outgoing edge (or only self-loops).
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]

    # If no clear sources (cyclic graph), fall back to all nodes as sources.
    if not sources:
        sources = list(G.nodes)

    chains: List[List[str]] = []
    for src in sources:
        for sink in sinks:
            if src == sink:
                continue
            for path in nx.all_simple_paths(G, src, sink):
                chains.append(list(path))
        # Also include length-1 chains for isolated sources that are also sinks.
        if G.out_degree(src) == 0:
            chains.append([src])

    # Deduplicate (a path may be a sub-path of another).
    # Keep only maximal paths — remove any path that is a contiguous
    # sub-sequence of a longer path.
    chains.sort(key=lambda c: (-len(c), c))
    maximal: List[List[str]] = []
    for chain in chains:
        is_sub = False
        chain_tuple = tuple(chain)
        for existing in maximal:
            existing_str = " ".join(existing)
            chain_str = " ".join(chain)
            if chain_str in existing_str:
                is_sub = True
                break
        if not is_sub:
            maximal.append(chain)

    if not include_none:
        stripped: List[List[str]] = []
        for chain in maximal:
            filtered = [n for n in chain if n != "NONE"]
            if filtered:
                stripped.append(filtered)
        maximal = stripped

    maximal.sort(key=lambda c: (-len(c), c))
    return maximal


def compute_path_depth(binding_edges: List[Tuple[str, str]]) -> int:
    G = nx.DiGraph()
    G.add_edges_from(binding_edges)
    if not G:
        return 0
    try:
        all_lengths = []
        for source in G.nodes:
            for target in G.nodes:
                if source != target:
                    for path in nx.all_simple_paths(G, source, target):
                        all_lengths.append(len(path) - 1)
        return max(all_lengths) if all_lengths else 0
    except Exception:
        return 0

def get_detail(load_statistics):
    '''
    Input: results loaded from jsonl file generated by llm_evaluation.py
    '''
    syntax_error = []
    execution_failed = []
    result_wrong = []
    pass_list = []
    for idx, item in enumerate(load_statistics['eval_result']):
        if 'result' in item and np.all(item['result']):
            pass_list.append(idx)
        else:
            if item["error"] is not None:
                syntax_error.append(idx)
                continue
            evaluator_return = item['result'][1]
            has_recorded = False
            result_pass = True
            for test_case_result in evaluator_return:
                # per test case
                if test_case_result["result_pass"] == False or test_case_result["state_pass"] == False:
                    if test_case_result["error_info"] is not None:
                        if "SyntaxError" in test_case_result["error_info"]:
                            syntax_error.append(idx)
                        else:
                            if "Traceback" in test_case_result["error_info"]:
                                execution_failed.append(idx)
                            else:
                                result_wrong.append(idx)
                        has_recorded = True
                        break
                    else:
                        if test_case_result["result_pass"] == False:
                            result_pass = False
                            continue
                else:
                    continue
            if not has_recorded:
                result_wrong.append(idx)

    assert len(syntax_error + result_wrong + execution_failed) == (40 - len(pass_list))
    return (syntax_error, execution_failed, result_wrong)