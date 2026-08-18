"""Tests for main module configuration."""

import ast
import os
import pathlib

from absl import logging
from absl.testing import absltest


class MainConfigTest(absltest.TestCase):

  def test_viewer_internal_locking_calls_are_outside_viewer_lock(self):
    main_module_path = pathlib.Path(__file__).parent / 'main.py'
    tree = ast.parse(main_module_path.read_text(encoding='utf-8'))
    internally_locked = {
        'clear_figures',
        'clear_images',
        'set_figures',
        'set_images',
        'set_texts',
        'sync',
    }

    nested_calls = []
    for node in ast.walk(tree):
      if not isinstance(node, ast.With):
        continue
      locks_viewer = any(
          isinstance(item.context_expr, ast.Call)
          and isinstance(item.context_expr.func, ast.Attribute)
          and item.context_expr.func.attr == 'lock'
          and isinstance(item.context_expr.func.value, ast.Name)
          and item.context_expr.func.value.id == 'viewer'
          for item in node.items
      )
      if not locks_viewer:
        continue
      nested_calls.extend(
          call.func.attr
          for call in ast.walk(node)
          if isinstance(call, ast.Call)
          and isinstance(call.func, ast.Attribute)
          and isinstance(call.func.value, ast.Name)
          and call.func.value.id == 'viewer'
          and call.func.attr in internally_locked
      )

    self.assertEmpty(
        nested_calls,
        'MuJoCo viewer APIs that lock internally must not run under '
        f'viewer.lock(): {nested_calls}',
    )

  def test_logging_verbosity_is_warning(self):
    """Test that logging verbosity in __main__.py is set to WARNING, not INFO.

    This test ensures that debug logging is not accidentally committed
    """
    # Get the path to __main__.py
    current_dir = pathlib.Path(__file__).parent
    main_module_path = current_dir / '__main__.py'

    # Make sure the file exists
    self.assertTrue(
        main_module_path.exists(),
        f'__main__.py not found at {main_module_path}',
    )

    # Parse the Python file and examine the AST
    with open(main_module_path, 'r') as f:
      source_code = f.read()

    tree = ast.parse(source_code)

    # Look for the line setting logging verbosity
    found_verbosity_setting = False
    correct_verbosity = False

    for node in ast.walk(tree):
      # Look for a call to logging.set_verbosity()
      if (
          isinstance(node, ast.Expr)
          and isinstance(node.value, ast.Call)
          and isinstance(node.value.func, ast.Attribute)
      ):

        # Check if it's logging.set_verbosity
        if (
            node.value.func.attr == 'set_verbosity'
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == 'logging'
        ):

          found_verbosity_setting = True

          # Check the argument to set_verbosity
          if len(node.value.args) > 0:
            arg = node.value.args[0]
            if (
                isinstance(arg, ast.Attribute)
                and arg.attr == 'WARNING'
                and isinstance(arg.value, ast.Name)
                and arg.value.id == 'logging'
            ):

              correct_verbosity = True
              break

    self.assertTrue(
        found_verbosity_setting,
        'Could not find logging.set_verbosity() call in __main__.py',
    )
    self.assertTrue(
        correct_verbosity,
        'Logging verbosity is not set to logging.WARNING. '
        "Make sure you're not accidentally committing debug logging settings!",
    )


if __name__ == '__main__':
  absltest.main()
