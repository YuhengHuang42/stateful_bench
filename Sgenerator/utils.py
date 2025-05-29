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
    changes = {}
    for k, v in new_book.items():
        if k not in old_book:
            # New key
            changes[k] = v
        else:
            assert v >= old_book[k], f"The number of occurence of {k} is decreased from {old_book[k]} to {v}."
            # Existing key with increased value
            if v > old_book[k]:
                changes[k] = v - old_book[k]
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
        self.var_def: Dict[str, Set[Tuple[str, int]]] = {}  # var_name -> set of (api_func, api_lineno) origins
        self.calls_set: Set[Tuple[Optional[str], Tuple[str, ...], str, int]] = set() # (assigned_var, sorted_args_tuple, func_name, lineno)
        self.control_stack_depth = 0
        self.control_gated_calls: Set[Tuple[str, int]] = set() # (func_name, lineno)
    
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
            # This might be too complex for a simple name, or could be an intermediate call
            # For now, let's represent it to see what comes out
            # parts.append(self._get_qualified_name(current.func) + "()") # Simplified
            pass # Avoid overly complex names like foo()().bar
        return '.'.join(reversed(parts)) if parts else '<unknown>'

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

        # Add origins from API calls on RHS
        # Also add these calls to self.calls_set
        primary_assigned_var = target_var_names[0] if target_var_names else None

        for sub_node in ast.walk(node.value):
            if isinstance(sub_node, ast.Call):
                func_name = self._get_qualified_name(sub_node.func)
                if not self._is_excluded(func_name):
                    current_origins.add((func_name, sub_node.lineno))
                    
                    args = self._get_arguments(sub_node)
                    args_tuple = tuple(sorted(list(set(args)))) # Ensure args are unique and sorted
                    
                    assigned_to_this_specific_call = None
                    if sub_node == node.value: # This call is the direct RHS
                         assigned_to_this_specific_call = primary_assigned_var
                    
                    call_tuple = (assigned_to_this_specific_call, args_tuple, func_name, sub_node.lineno)
                    self.calls_set.add(call_tuple)

                    if self.control_stack_depth > 0:
                        self.control_gated_calls.add((func_name, sub_node.lineno))
        
        # Add origins propagated from variables referenced on RHS
        referenced_vars_in_rhs = self._collect_variable_refs(node.value)
        for ref_var in referenced_vars_in_rhs:
            current_origins.update(self.var_def.get(ref_var, set()))

        for var_name in target_var_names:
            self.var_def[var_name] = current_origins.copy()
            
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            func_name = self._get_qualified_name(node.value.func)
            if not self._is_excluded(func_name):
                args = self._get_arguments(node.value)
                args_tuple = tuple(sorted(list(set(args))))
                call_tuple = (None, args_tuple, func_name, node.value.lineno)
                self.calls_set.add(call_tuple)
                if self.control_stack_depth > 0:
                    self.control_gated_calls.add((func_name, node.value.lineno))
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        iterable_origins: Set[Tuple[str, int]] = set()
        iterable_referenced_vars = self._collect_variable_refs(node.iter)
        for ref_var in iterable_referenced_vars:
            iterable_origins.update(self.var_def.get(ref_var, set()))
        
        # If the iterable itself is an API call
        if isinstance(node.iter, ast.Call):
            func_name = self._get_qualified_name(node.iter.func)
            if not self._is_excluded(func_name):
                iterable_origins.add((func_name, node.iter.lineno))
                # This call should also be in self.calls_set if not already added by visit_Assign/Expr
                args = self._get_arguments(node.iter)
                args_tuple = tuple(sorted(list(set(args))))
                # Determine if this call was assigned. If node.iter is part of an assignment's RHS,
                # visit_Assign would handle it. This adds it if it's solely an iterable.
                # This logic can be complex; for now, assume it's not assigned here.
                self.calls_set.add((None, args_tuple, func_name, node.iter.lineno))
                if self.control_stack_depth > 0: # Depth check for call within for-loop header
                    self.control_gated_calls.add((func_name, node.iter.lineno))


        loop_vars: List[str] = []
        if isinstance(node.target, ast.Name):
            loop_vars.append(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    loop_vars.append(elt.id)
        
        for loop_var in loop_vars:
            # Initialize or update origins. If loop_var is reassigned inside, visit_Assign handles that.
            # This sets initial taint for its use within the loop.
            self.var_def[loop_var] = self.var_def.get(loop_var, set()).union(iterable_origins)

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
                # Propagate origins from item.context_expr to item.optional_vars
                context_origins: Set[Tuple[str, int]] = set()
                
                # Origins from API calls in context_expr
                for sub_node in ast.walk(item.context_expr):
                    if isinstance(sub_node, ast.Call):
                        func_name = self._get_qualified_name(sub_node.func)
                        if not self._is_excluded(func_name):
                            context_origins.add((func_name, sub_node.lineno))
                            # Add this call to self.calls_set
                            args = self._get_arguments(sub_node)
                            args_tuple = tuple(sorted(list(set(args))))
                            # If item.context_expr == sub_node, it's assigned to item.optional_vars.id
                            assigned_to = item.optional_vars.id if isinstance(item.context_expr, ast.Call) and self._get_qualified_name(item.context_expr.func) == func_name and item.context_expr.lineno == sub_node.lineno else None
                            self.calls_set.add((assigned_to, args_tuple, func_name, sub_node.lineno))
                            if self.control_stack_depth > 0:
                                self.control_gated_calls.add((func_name, sub_node.lineno))
                
                # Origins from referenced vars in context_expr
                ref_vars = self._collect_variable_refs(item.context_expr)
                for ref_var in ref_vars:
                    context_origins.update(self.var_def.get(ref_var, set()))
                
                self.var_def[item.optional_vars.id] = context_origins
                
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