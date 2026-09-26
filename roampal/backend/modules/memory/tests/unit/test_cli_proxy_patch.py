"""Task 25 / Item 8: patch("roampal.cli.X") rebinds X into EVERY
roampal.cli.* module holding the same object and restores all of them.
From-imports mean one name can be held by several modules (``GREEN`` in
``_common``, ``memory_cmds``, ``setup``, ``update_check``); a single-owner
rebind would leave the rest seeing the original."""

import unittest
from unittest import mock

import roampal.cli


class TestPatchRebindsEveryHolder(unittest.TestCase):
    def test_green_is_shared(self):
        # precondition for the patch tests: from-imports really do share
        import roampal.cli._common as common
        import roampal.cli.memory_cmds as memory_cmds
        import roampal.cli.setup as setup
        import roampal.cli.update_check as update_check

        holders = {common.GREEN, memory_cmds.GREEN, setup.GREEN, update_check.GREEN}
        self.assertEqual(len(holders), 1, "precondition broke: holders diverged")

    def test_patch_rebinds_all_holders_and_restores(self):
        import roampal.cli._common as common
        import roampal.cli.memory_cmds as memory_cmds
        import roampal.cli.setup as setup
        import roampal.cli.update_check as update_check

        original = common.GREEN
        with mock.patch("roampal.cli.GREEN", "PATCHED"):
            for mod in (common, memory_cmds, setup, update_check):
                self.assertIs(mod.GREEN, "PATCHED", mod.__name__)
        for mod in (common, memory_cmds, setup, update_check):
            self.assertIs(mod.GREEN, original, mod.__name__)

    def test_nested_patches_restore_each_layer(self):
        with mock.patch("roampal.cli.GREEN", "OUTER"):
            import roampal.cli._common as common
            self.assertIs(common.GREEN, "OUTER")
            with mock.patch("roampal.cli.GREEN", "INNER"):
                import roampal.cli.setup as setup
                self.assertIs(setup.GREEN, "INNER")
            import roampal.cli.memory_cmds as memory_cmds
            self.assertIs(memory_cmds.GREEN, "OUTER")
        import roampal.cli.update_check as update_check
        restored = (common.GREEN, memory_cmds.GREEN, update_check.GREEN)
        for v in restored:
            self.assertNotIn(v, ("OUTER", "INNER"))


if __name__ == "__main__":
    unittest.main()
