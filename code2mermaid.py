#!/usr/bin/env python3
import sys
import ast
import argparse

class FlowchartGenerator(ast.NodeVisitor):
    """
    A flowchart generator that ensures each decision node
    has exactly two outflows (Yes/No) and merges the branches
    back together. Also handles try/except, returns, breaks, etc.,
    including loop-else clauses.

    This version replaces problematic characters like { } [ ] "
    with HTML entities to avoid Mermaid parse issues.
    """

    def __init__(self):
        self.node_map = {}
        self.edges = []
        self.node_count = 0

        # Track loop contexts (cond_node, merge_node)
        # so 'break' and 'continue' can jump properly.
        self.loop_stack = []

        # Track the current function's end node for 'return'
        self.current_function_end = None

    def _escape_label(self, label: str) -> str:
        """
        Convert special characters into HTML entities so that Mermaid
        doesn't parse them as shape syntax or string terminators.
        """
        replacements = {
            "&":  "&amp;",
            "<":  "&lt;",
            ">":  "&gt;",
            '"':  "&quot;",
            "[":  "&#91;",
            "]":  "&#93;",
            "{":  "&#123;",
            "}":  "&#125;",
        }
        for old, new in replacements.items():
            label = label.replace(old, new)
        return label

    def _new_node(self, shape: str, label: str) -> str:
        """
        Create a Mermaid node with a specific shape:
          - 'start_end' => ((label)) (oval)
          - 'decision'  => {label}   (diamond)
          - 'input_output' => [/"label"/] (parallelogram)
          - else => ["label"]        (rectangle)
        """
        node_id = f"node{self.node_count + 1}"
        self.node_count += 1

        label_escaped = self._escape_label(label)

        if shape == "input_output":
            definition = f'{node_id}[/"{label_escaped}"/]'
        elif shape == "start_end":
            definition = f'{node_id}(({label_escaped}))'
        elif shape == "decision":
            definition = f'{node_id}{{"{label_escaped}"}}'
        else:
            definition = f'{node_id}["{label_escaped}"]'

        self.node_map[node_id] = definition
        return node_id

    def _link(self, src: str, dest: str, label: str = None):
        """
        Create an edge: src --> dest
        or if label is "Yes"/"No", src -- Yes --> dest
        """
        if dest is None:
            return  # No link if destination is None
        if label in ("Yes", "No"):
            self.edges.append(f"{src} -- {label} --> {dest}")
        else:
            self.edges.append(f"{src} --> {dest}")

    def generate_mermaid(self) -> str:
        lines = ["graph TD"]
        for definition in self.node_map.values():
            lines.append("    " + definition)
        for edge in self.edges:
            lines.append("    " + edge)
        return "\n".join(lines)

    #
    # == Top-Level Visitors ==
    #
    def visit_Module(self, node):
        # If you have multiple top-level statements, handle them
        self.generic_visit(node)

    def visit_FunctionDef(self, func_node):
        """
        For each function:
          - Start node, chain statements, end node
          - Keep track of a function_end node for 'return' statements
        """
        func_name = func_node.name
        start_id = self._new_node("start_end", f"Start: {func_name}")
        end_id   = self._new_node("start_end", f"End: {func_name}")

        saved_end = self.current_function_end
        self.current_function_end = end_id

        prev_node = start_id
        for stmt in func_node.body:
            first_node, last_node = self._build_flow(stmt)
            self._link(prev_node, first_node)
            if last_node is None:
                # short-circuit
                break
            prev_node = last_node

        # If flow never short-circuited, link to function end
        self._link(prev_node, end_id)

        self.current_function_end = saved_end

    #
    # == The main dispatcher ==
    #
    def _build_flow(self, node):
        if isinstance(node, ast.Try):
            return self._handle_try(node)
        elif isinstance(node, ast.Assign):
            return self._handle_assign(node)
        elif isinstance(node, ast.If):
            return self._handle_if(node)
        elif isinstance(node, ast.For):
            return self._handle_for(node)
        elif isinstance(node, ast.While):
            return self._handle_while(node)
        elif isinstance(node, ast.Expr):
            return self._handle_expr(node)
        elif isinstance(node, ast.FunctionDef):
            return self._handle_functiondef(node)
        elif isinstance(node, ast.Return):
            return self._handle_return(node)
        elif isinstance(node, ast.Break):
            return self._handle_break(node)
        elif isinstance(node, ast.Continue):
            return self._handle_continue(node)
        elif isinstance(node, ast.Raise):
            return self._handle_raise(node)
        elif isinstance(node, ast.Pass):
            return self._handle_pass(node)
        else:
            # fallback
            dump_str = ast.dump(node)
            unk_id = self._new_node("process", f"Unsupported: {dump_str}")
            return (unk_id, unk_id)

    def _build_block(self, stmts):
        """
        Build a sub-flow for multiple statements in sequence.
        Return (first_node_id, last_node_id).
        """
        if not stmts:
            dummy_id = self._new_node("process", "pass")
            return (dummy_id, dummy_id)

        first_node, last_node = self._build_flow(stmts[0])
        prev = last_node
        for stmt in stmts[1:]:
            if prev is None:
                break  # short-circuited
            stmt_first, stmt_last = self._build_flow(stmt)
            self._link(prev, stmt_first)
            prev = stmt_last
        return (first_node, prev)

    #
    # == Statement Handlers ==
    #
    def _handle_try(self, node: ast.Try):
        try_id = self._new_node("process", "Try")

        # Try block
        if node.body:
            first_try, last_try = self._build_block(node.body)
            self._link(try_id, first_try, label="Enter")
        else:
            dummy_try = self._new_node("process", "pass")
            self._link(try_id, dummy_try)
            last_try = dummy_try

        normal_end_nodes = [last_try] if last_try is not None else []

        # Except handlers
        except_end_nodes = []
        for handler in node.handlers:
            if handler.type:
                exc_label = f"Except {ast.unparse(handler.type)}"
            else:
                exc_label = "Except (any)"
            if handler.name:
                exc_label += f" as {handler.name}"

            except_id = self._new_node("process", exc_label)
            self._link(try_id, except_id, label="Exception?")
            first_exc, last_exc = self._build_block(handler.body)
            self._link(except_id, first_exc)
            if last_exc is not None:
                except_end_nodes.append(last_exc)

        # Else
        else_end_node = None
        if node.orelse:
            else_id = self._new_node("process", "Else")
            for n in normal_end_nodes:
                self._link(n, else_id)
            first_else, last_else = self._build_block(node.orelse)
            self._link(else_id, first_else)
            else_end_node = last_else
        else:
            else_end_node = normal_end_nodes[0] if normal_end_nodes else None

        # Finally
        finally_end_node = None
        if node.finalbody:
            fin_id = self._new_node("process", "Finally")
            first_fin, last_fin = self._build_block(node.finalbody)

            if else_end_node is not None:
                self._link(else_end_node, fin_id)
            else:
                for n in normal_end_nodes:
                    self._link(n, fin_id)

            for eend in except_end_nodes:
                self._link(eend, fin_id)

            self._link(fin_id, first_fin)
            finally_end_node = last_fin

        subflow_first = try_id
        if node.finalbody:
            subflow_last = finally_end_node
        else:
            subflow_last = else_end_node

        return (subflow_first, subflow_last)

    def _handle_assign(self, node: ast.Assign):
        target_str = ast.unparse(node.targets[0])
        value_str  = ast.unparse(node.value)
        label = f"{target_str} = {value_str}"

        # If it's input() => use I/O shape
        if (isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "input"):
            nid = self._new_node("input_output", label)
        else:
            nid = self._new_node("process", label)
        return (nid, nid)

    def _handle_if(self, if_node: ast.If):
        cond_id = self._new_node("decision", f"If {ast.unparse(if_node.test)}?")

        # If body
        if if_node.body:
            if_body_first, if_body_last = self._build_block(if_node.body)
        else:
            if_body_first, if_body_last = self._make_pass()
        self._link(cond_id, if_body_first, label="Yes")

        # Else body
        if if_node.orelse:
            else_body_first, else_body_last = self._build_block(if_node.orelse)
        else:
            else_body_first, else_body_last = self._make_pass()
        self._link(cond_id, else_body_first, label="No")

        # Merge
        merge_id = self._new_node("process", "merge")
        if if_body_last is not None:
            self._link(if_body_last, merge_id)
        if else_body_last is not None:
            self._link(else_body_last, merge_id)

        return (cond_id, merge_id)

    def _make_pass(self):
        n = self._new_node("process", "pass")
        return (n, n)

    def _handle_for(self, for_node: ast.For):
        """
        Handle a 'for' loop, including optional else-block,
        with exactly two arrows from the diamond:
          - Yes => loop body
          - No => either else-block or merge node
        """
        cond_label = f"For {ast.unparse(for_node.target)} in {ast.unparse(for_node.iter)}?"
        cond_id = self._new_node("decision", cond_label)
        merge_id = self._new_node("process", "for-merge")

        # Push loop context so 'break' goes to merge_id, 'continue' goes to cond_id
        self.loop_stack.append((cond_id, merge_id))

        # Build the loop body
        if for_node.body:
            body_first, body_last = self._build_block(for_node.body)
        else:
            body_first, body_last = self._make_pass()

        # Link diamond => body, body => diamond
        self._link(cond_id, body_first, label="Yes")
        self._link(body_last, cond_id)

        # Build the else block (if any) from the diamond's No edge
        if for_node.orelse:
            else_first, else_last = self._build_block(for_node.orelse)
            self._link(cond_id, else_first, label="No")
            # Then link the end of else block to the merge
            if else_last is not None:
                self._link(else_last, merge_id)
        else:
            # If no else block, go straight to merge on No
            self._link(cond_id, merge_id, label="No")

        self.loop_stack.pop()
        return (cond_id, merge_id)

    def _handle_while(self, while_node: ast.While):
        """
        Handle a 'while' loop, including optional else-block,
        with exactly two arrows from the diamond:
          - Yes => loop body
          - No => either else-block or merge node
        """
        cond_label = f"While {ast.unparse(while_node.test)}?"
        cond_id = self._new_node("decision", cond_label)
        merge_id = self._new_node("process", "while-merge")

        # Push loop context so 'break' goes to merge_id, 'continue' goes to cond_id
        self.loop_stack.append((cond_id, merge_id))

        # Build the loop body
        if while_node.body:
            body_first, body_last = self._build_block(while_node.body)
        else:
            body_first, body_last = self._make_pass()

        # Link diamond => body, body => diamond
        self._link(cond_id, body_first, label="Yes")
        self._link(body_last, cond_id)

        # Build the else block
        if while_node.orelse:
            else_first, else_last = self._build_block(while_node.orelse)
            self._link(cond_id, else_first, label="No")
            if else_last is not None:
                self._link(else_last, merge_id)
        else:
            self._link(cond_id, merge_id, label="No")

        self.loop_stack.pop()
        return (cond_id, merge_id)

    def _handle_expr(self, expr_node: ast.Expr):
        if isinstance(expr_node.value, ast.Call):
            call_node = expr_node.value
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
                args_str  = ", ".join(ast.unparse(a) for a in call_node.args)
                label     = f"{func_name}({args_str})"
                if func_name == "print":
                    nid = self._new_node("input_output", label)
                    return (nid, nid)
        code_str = ast.unparse(expr_node)
        nid = self._new_node("process", code_str)
        return (nid, nid)

    def _handle_functiondef(self, func_node: ast.FunctionDef):
        top_id = self._new_node("process", f"FunctionDef(name='{func_node.name}')")

        start_id = self._new_node("start_end", f"Start func {func_node.name}")
        end_id   = self._new_node("start_end", f"End func {func_node.name}")
        self._link(top_id, start_id)

        saved_end = self.current_function_end
        self.current_function_end = end_id

        prev = start_id
        for stmt in func_node.body:
            f1, f2 = self._build_flow(stmt)
            self._link(prev, f1)
            if f2 is None:
                break
            prev = f2
        self._link(prev, end_id)

        self.current_function_end = saved_end
        return (top_id, top_id)

    #
    # == Short-Circuit Handlers (return/break/continue/raise) ==
    #
    def _handle_return(self, node: ast.Return):
        label = "Return"
        if node.value:
            label += f" {ast.unparse(node.value)}"
        r_id = self._new_node("process", label)

        # Link to function end if known
        if self.current_function_end:
            self._link(r_id, self.current_function_end)

        # short-circuit
        return (r_id, None)

    def _handle_break(self, node: ast.Break):
        b_id = self._new_node("process", "break")
        if self.loop_stack:
            cond_node, merge_node = self.loop_stack[-1]
            self._link(b_id, merge_node)  # jump to loop merge
        return (b_id, None)

    def _handle_continue(self, node: ast.Continue):
        c_id = self._new_node("process", "continue")
        if self.loop_stack:
            cond_node, merge_node = self.loop_stack[-1]
            self._link(c_id, cond_node)  # jump back to condition
        return (c_id, None)

    def _handle_raise(self, node: ast.Raise):
        if node.exc:
            label = f"Raise {ast.unparse(node.exc)}"
        else:
            label = "Raise"
        r_id = self._new_node("process", label)
        return (r_id, None)

    def _handle_pass(self, node: ast.Pass):
        p_id = self._new_node("process", "pass")
        return (p_id, p_id)

#
# A helper to parse code and generate the flowchart
#
def generate_flowchart_from_code(code: str) -> str:
    tree = ast.parse(code)
    gen = FlowchartGenerator()
    gen.visit(tree)
    return gen.generate_mermaid()

#
# CLI
#
def main():
    parser = argparse.ArgumentParser(
        description="Generate a structured Mermaid flowchart, converting special characters into HTML entities."
    )
    parser.add_argument("source_file", help="Path to the Python source file.")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional path to save the Mermaid output (e.g. .mmd). If not given, prints to stdout."
    )
    args = parser.parse_args()

    # Read the source file
    try:
        with open(args.source_file, "r", encoding="utf-8") as f:
            code = f.read()
    except IOError:
        print(f"Error: Could not read file {args.source_file}", file=sys.stderr)
        sys.exit(1)

    # Generate flowchart
    try:
        mermaid_code = generate_flowchart_from_code(code)
    except SyntaxError as e:
        print(f"Syntax Error in Python code: {e}", file=sys.stderr)
        sys.exit(2)

    # Output
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as out_f:
                out_f.write(mermaid_code + "\n")
        except IOError:
            print(f"Error: Could not write to file {args.output}", file=sys.stderr)
            sys.exit(3)
    else:
        print(mermaid_code)

if __name__ == "__main__":
    main()
