#!/usr/bin/env python3

r'''###############################################################################
###################################################################################
#
#	Docs Utils Python module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#	https://github.com/asigalov61/tegridy-tools
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
'''

###################################################################################

from __future__ import annotations

import sys
import ast
import io
import os
import shutil
import re
import textwrap

from typing import Dict, Optional, Union, List, Set, Tuple

###################################################################################

# ---------- Utilities for source extraction ----------

def _get_source_segment(source: str, node: ast.AST) -> str:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    lines = source.splitlines()
    start_line = node.lineno - 1
    end_line = node.end_lineno - 1
    if start_line == end_line:
        return lines[start_line][node.col_offset: node.end_col_offset]
    parts = []
    parts.append(lines[start_line][node.col_offset:])
    for i in range(start_line + 1, end_line):
        parts.append(lines[i])
    parts.append(lines[end_line][: node.end_col_offset])
    return "\n".join(parts)

# ---------- AST visitors to collect names ----------

class NameCollector(ast.NodeVisitor):
    def __init__(self):
        self.loads: Set[str] = set()
        self.stores: Set[str] = set()
        self.attrs: Set[str] = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Param)):
            self.stores.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        value = node.value
        if isinstance(value, ast.Name):
            self.attrs.add(value.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # nested function name is a store; do not traverse nested body
        self.stores.add(node.name)
        for deco in node.decorator_list:
            self.visit(deco)
        for arg in getattr(node.args, "args", []):
            if arg.arg:
                self.stores.add(arg.arg)
        # skip body to avoid mixing nested helper references with outer function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

# ---------- Module analysis ----------

def analyze_module(source: str) -> Dict[str, object]:
    tree = ast.parse(source)
    functions: Dict[str, ast.FunctionDef] = {}
    assigns: Dict[str, ast.AST] = {}
    imports: List[ast.AST] = []
    other_top_level_names: Set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t.id)
            else:
                t = node.target
                if isinstance(t, ast.Name):
                    targets.append(t.id)
            for name in targets:
                assigns[name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            other_top_level_names.add(node.name)
    return {
        "tree": tree,
        "functions": functions,
        "assigns": assigns,
        "imports": imports,
        "other_names": other_top_level_names,
    }

def collect_references_for_function(func_node: ast.FunctionDef) -> Tuple[Set[str], Set[str]]:
    collector = NameCollector()
    for stmt in func_node.body:
        collector.visit(stmt)
    param_names = {arg.arg for arg in func_node.args.args}
    loads = collector.loads - collector.stores - param_names
    return loads, collector.attrs

def build_dependency_graph(analysis: dict) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    functions: Dict[str, ast.FunctionDef] = analysis["functions"]
    assigns: Dict[str, ast.AST] = analysis["assigns"]
    imports: List[ast.AST] = analysis["imports"]
    other_names: Set[str] = analysis["other_names"]

    func_deps: Dict[str, Set[str]] = {}
    const_deps: Dict[str, Set[str]] = {}
    import_deps: Dict[str, Set[str]] = {}

    imported_names: Set[str] = set()
    for imp in imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(imp, ast.ImportFrom):
            for alias in imp.names:
                imported_names.add(alias.asname or alias.name)

    for fname, fnode in functions.items():
        loads, attrs = collect_references_for_function(fnode)
        fdeps = set()
        cdeps = set()
        ideps = set()
        for name in loads | attrs:
            if name in functions and name != fname:
                fdeps.add(name)
            elif name in assigns:
                cdeps.add(name)
            elif name in imported_names:
                ideps.add(name)
        func_deps[fname] = fdeps
        const_deps[fname] = cdeps
        import_deps[fname] = ideps

    return func_deps, const_deps, import_deps

# ---------- Topological sort using Kahn's algorithm ----------

def topo_sort_subset_kahn(nodes: Set[str], deps: Dict[str, Set[str]]) -> Tuple[List[str], Optional[List[str]]]:
    in_degree: Dict[str, int] = {n: 0 for n in nodes}
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for n in nodes:
        for d in deps.get(n, set()):
            if d in nodes:
                in_degree[n] += 1
                adj[d].add(n)

    queue: List[str] = [n for n, deg in in_degree.items() if deg == 0]
    ordered: List[str] = []
    while queue:
        n = queue.pop(0)
        ordered.append(n)
        for dependent in sorted(adj.get(n, set())):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(nodes):
        cycle_nodes = [n for n, deg in in_degree.items() if deg > 0]
        return ordered, cycle_nodes
    return ordered, None

# ---------- Gathering source pieces ----------

def gather_import_source(analysis: dict, source: str, needed_import_names: Set[str]) -> List[str]:
    imports_src: List[str] = []
    for imp in analysis["imports"]:
        names = []
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                names.append(alias.asname or alias.name.split(".")[0])
        elif isinstance(imp, ast.ImportFrom):
            for alias in imp.names:
                names.append(alias.asname or alias.name)
        if any(n in needed_import_names for n in names):
            imports_src.append(_get_source_segment(source, imp))
    return imports_src

def gather_assign_source(analysis: dict, source: str, needed_assigns: List[str]) -> List[str]:
    assigns_src: List[str] = []
    for name in needed_assigns:
        node = analysis["assigns"].get(name)
        if node is not None:
            assigns_src.append(_get_source_segment(source, node))
    return assigns_src

def gather_function_source(analysis: dict, source: str, ordered_funcs: List[str]) -> List[str]:
    funcs_src: List[str] = []
    for name in ordered_funcs:
        node = analysis["functions"].get(name)
        if node is not None:
            funcs_src.append(_get_source_segment(source, node))
    return funcs_src

# ---------- Compose content for a primary function ----------

def compose_content_for_primary(primary: str,
                                analysis: dict,
                                source: str,
                                func_deps: Dict[str, Set[str]],
                                const_deps: Dict[str, Set[str]],
                                import_deps: Dict[str, Set[str]]) -> Tuple[str, Optional[List[str]]]:
    """
    Build the text content for a single primary function.
    Returns (content, cycle_nodes_or_None).
    """
    # compute closure
    to_visit = [primary]
    closure_funcs: Set[str] = set()
    closure_consts: Set[str] = set()
    closure_imports: Set[str] = set()
    while to_visit:
        cur = to_visit.pop()
        if cur in closure_funcs:
            continue
        closure_funcs.add(cur)
        for fdep in func_deps.get(cur, set()):
            if fdep not in closure_funcs:
                to_visit.append(fdep)
        for cdep in const_deps.get(cur, set()):
            closure_consts.add(cdep)
        for idep in import_deps.get(cur, set()):
            closure_imports.add(idep)

    ordered_funcs, cycle = topo_sort_subset_kahn(closure_funcs, func_deps)
    if cycle:
        for c in sorted(cycle):
            if c not in ordered_funcs:
                ordered_funcs.append(c)

    imports_src = gather_import_source(analysis, source, closure_imports)

    assigns_ordered: List[str] = []
    for node in analysis["tree"].body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t.id)
            else:
                t = node.target
                if isinstance(t, ast.Name):
                    targets.append(t.id)
            for tname in targets:
                if tname in closure_consts and tname not in assigns_ordered:
                    assigns_ordered.append(tname)

    assigns_src = gather_assign_source(analysis, source, assigns_ordered)
    funcs_src = gather_function_source(analysis, source, ordered_funcs)

    parts: List[str] = []
    #parts.append(f"# Exported from module: {os.path.basename(getattr(analysis['tree'], 'filename', '<source>'))}")
    #parts.append(f"# Primary function: {primary}")
    if cycle:
        parts.append(f"# NOTE: Circular dependency detected among: {', '.join(sorted(cycle))}")
    if imports_src:
        parts.append("\n".join(imports_src))
    if assigns_src:
        parts.append("\n\n".join(assigns_src))
    if funcs_src:
        parts.append("\n\n".join(funcs_src))
    content = "\n\n".join(parts).rstrip() + "\n"
    return content, cycle

# ---------- Main dumping logic (public) ----------

def dump_functions_with_deps(module_path: str,
                             out_dir: Optional[str] = None,
                             return_as_dict: bool = False) -> Optional[Dict[str, str]]:
    """
    Parse module_path and produce per-primary-function text outputs.

    Parameters:
      - module_path: path to the .py module to analyze
      - out_dir: directory to write .txt files; if None, no files are written
      - return_as_dict: if True, return a dict {primary_func_name: content}

    Returns:
      - dict mapping primary function names to content if return_as_dict=True, else None.
    """
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(module_path, "r", encoding="utf-8") as f:
        source = f.read()

    analysis = analyze_module(source)
    # attach filename to AST tree for compose header
    setattr(analysis["tree"], "filename", os.path.basename(module_path))
    functions = analysis["functions"]
    if not functions:
        print("No top-level functions found in module.")
        return {} if return_as_dict else None

    func_deps, const_deps, import_deps = build_dependency_graph(analysis)

    results: Dict[str, str] = {}
    for primary in functions.keys():
        content, cycle = compose_content_for_primary(primary, analysis, source, func_deps, const_deps, import_deps)
        results[primary] = content
        if out_dir:
            out_path = os.path.join(out_dir, f"{primary}.txt")
            with open(out_path, "w", encoding="utf-8") as outf:
                outf.write(content)

    if return_as_dict:
        return results
    return None

###################################################################################

def _clean_docstring(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    # Strip an outer quoted literal if the user passed the quotes
    m_outer = re.match(r'^\s*([rubfRUBF]*)(?P<q>["\']{3})(?P<body>.*)(?P=q)\s*$', s, re.DOTALL)
    if m_outer:
        s = m_outer.group('body')
    # Strip inner prefixed triple-quoted block like r'''...''' if present
    m_inner = re.match(r'^\s*([rubfRUBF]+)\s*(?P<q>["\']{3})(?P<body>.*)(?P=q)\s*$', s, re.DOTALL)
    if m_inner:
        s = m_inner.group('body')
    s = textwrap.dedent(s)
    s = s.strip('\n').strip()
    return s or None


def _build_parent_map(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def _node_key(node: Union[ast.FunctionDef, ast.AsyncFunctionDef], parent_map: Dict[ast.AST, ast.AST]) -> str:
    p = parent_map.get(node)
    if isinstance(p, ast.ClassDef):
        return f"{p.name}.{node.name}"
    return node.name


def _doc_node_range(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[Tuple[int, int]]:
    if not node.body:
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        start = first.lineno - 1
        end = getattr(first, "end_lineno", first.lineno) - 1
        return start, end
    return None


def _choose_quote_and_escape(doc: str) -> Tuple[str, str]:
    triple_double = '"""'
    triple_single = "'''"
    if triple_double not in doc:
        return triple_double, doc
    if triple_single not in doc:
        return triple_single, doc
    # both appear: escape triple-double sequences
    escaped = doc.replace(triple_double, '\\"""')
    return triple_double, escaped


def _format_doc_lines(doc: str, indent: str) -> List[str]:
    quote, safe_doc = _choose_quote_and_escape(doc)
    if '\n' not in safe_doc and len(safe_doc) <= 120:
        inner = safe_doc.replace(quote, quote[0] * 3)
        return [f'{indent}{quote}{inner}{quote}']
    lines = [f'{indent}{quote}']
    for ln in safe_doc.splitlines():
        lines.append(f'{indent}{ln}')
    lines.append(f'{indent}{quote}')
    return lines


def inject_docstrings_into_module(module_or_path: Union[object, str],
                                  doc_map: Dict[str, Optional[str]],
                                  backup: bool = True,
                                  encoding: str = "utf-8",
                                  update_runtime: bool = True) -> str:
    """
    Inject doc strings into a Python module
    """
    
    
    module_obj = None
    if isinstance(module_or_path, str):
        path = module_or_path
    else:
        module_obj = module_or_path
        path = getattr(module_or_path, "__file__", None)
        if path is None:
            raise ValueError("Module object has no __file__; pass a file path instead.")
    if not path.endswith(".py"):
        raise ValueError("Path must point to a .py source file.")

    with io.open(path, "r", encoding=encoding) as f:
        source = f.read()
    source_lines = source.splitlines()

    tree = ast.parse(source, filename=path)
    parent_map = _build_parent_map(tree)

    normalized_map: Dict[str, Optional[str]] = {k: _clean_docstring(v) for k, v in doc_map.items()}

    # Collect target function nodes
    targets: List[Tuple[int, ast.AST, Optional[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = _node_key(node, parent_map)
            if key in normalized_map:
                targets.append((node.lineno, node, normalized_map[key]))

    # Sort descending by lineno so edits don't shift later indices
    targets.sort(key=lambda t: t[0], reverse=True)

    for _, node, new_doc in targets:
        doc_range = _doc_node_range(node)

        # Determine indent: prefer indentation of first body statement; otherwise indent one level inside def
        indent = None
        if node.body:
            first_stmt = node.body[0]
            col = getattr(first_stmt, "col_offset", None)
            if col is not None:
                indent = " " * col
        if indent is None:
            indent = " " * (node.col_offset + 4)

        if doc_range:
            # Replace or remove existing docstring (AST-detected)
            start_idx, end_idx = doc_range
            if new_doc is None:
                del source_lines[start_idx:end_idx + 1]
            else:
                new_lines = _format_doc_lines(new_doc, indent)
                source_lines[start_idx:end_idx + 1] = new_lines
        else:
            # Insert BEFORE the first body statement (strictly), so docstring is the first statement.
            if not node.body:
                # nothing to insert into
                continue
            insert_at = node.body[0].lineno - 1  # strictly before the first statement
            if new_doc is None:
                continue
            new_lines = _format_doc_lines(new_doc, indent)
            source_lines[insert_at:insert_at] = new_lines
            # Ensure a single blank line after docstring for readability if not already blank
            after_idx = insert_at + len(new_lines)
            if after_idx < len(source_lines) and source_lines[after_idx].strip() != "":
                source_lines.insert(after_idx, "")

    final_source = "\n".join(source_lines)
    if not final_source.endswith("\n"):
        final_source += "\n"

    if backup:
        shutil.copy2(path, path + ".bak")

    tmp_path = path + ".tmp"
    with io.open(tmp_path, "w", encoding=encoding) as f:
        f.write(final_source)
    os.replace(tmp_path, path)

    # Update runtime __doc__ if requested and module object provided
    if update_runtime and module_obj is not None:
        for _, node, new_doc in targets:
            key = _node_key(node, parent_map)
            desired = normalized_map.get(key, normalized_map.get(node.name))
            parent = parent_map.get(node)
            try:
                if isinstance(parent, ast.ClassDef):
                    cls = getattr(module_obj, parent.name, None)
                    if cls is not None:
                        func = getattr(cls, node.name, None)
                        if func is not None:
                            func.__doc__ = desired
                else:
                    func = getattr(module_obj, node.name, None)
                    if func is not None:
                        func.__doc__ = desired
            except Exception:
                pass

    return path

###################################################################################

def generate_advanced_readme(strings, title, subtitle):
    """
    Generate a polished GitHub‑compatible README.md with:
    - Title + subtitle
    - Automatic table of contents
    - Alphabetical sections
    - Sorted entries
    - Collapsible sections
    - Letter counts
    - Clean spacing + horizontal rules

    Parameters
    ----------
    strings : list[str]
        List of strings to organize.
    title : str
        H1 title for README.
    subtitle : str
        H2 subtitle for README.

    Returns
    -------
    str
        A fully formatted README.md string.
    """

    # Normalize + sort
    cleaned = sorted(s.strip() for s in strings if s.strip())

    # Bucket by first letter
    sections = {}
    for s in cleaned:
        first = s[0].upper()
        if not first.isalpha():
            first = "#"  # Non-alphabetic bucket
        sections.setdefault(first, []).append(s)

    # Build README
    lines = []

    # Title + subtitle
    lines.append(f"# {title}")
    lines.append(f"## {subtitle}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table of Contents
    lines.append("## 📚 Table of Contents")
    lines.append("")
    for letter in sorted(sections.keys()):
        anchor = letter.lower()
        lines.append(f"- [{letter}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Alphabetical sections
    for letter in sorted(sections.keys()):
        items = sections[letter]
        anchor = letter.lower()

        lines.append(f"## {letter}")
        lines.append(f"<a name=\"{anchor}\"></a>")
        lines.append("")
        lines.append(f"**{len(items)} entr{'y' if len(items)==1 else 'ies'}**")
        lines.append("")

        # Collapsible block
        lines.append("<details>")
        lines.append("<summary>Show entries</summary>")
        lines.append("")
        for item in items:
            lines.append(f"* `{item}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines) + "\n"

###################################################################################
# This is the end of the Docs Utils Python module
###################################################################################