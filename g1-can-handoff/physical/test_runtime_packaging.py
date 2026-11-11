"""The deployed Stagefile must carry the runtime's local Python dependencies."""
import ast
from pathlib import Path
import re
import unittest


class RuntimePackagingTests(unittest.TestCase):
    def test_stagefile_copies_every_transitive_local_runtime_module(self):
        root=Path(__file__).resolve().parent
        descriptor=(root/'build.stagefile.yaml').read_text()
        copied={item.strip().strip('"\'')
                for group in re.findall(r'paths:\s*\[([^\]]+)\]',descriptor)
                for item in group.split(',')}
        pending=['runtime'];visited=set()
        while pending:
            name=pending.pop()
            if name in visited:continue
            visited.add(name)
            filename=name+'.py'
            self.assertIn(filename,copied,'Runtime dependency missing from Stagefile: '+filename)
            tree=ast.parse((root/filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):imports=[alias.name for alias in node.names]
                elif isinstance(node,ast.ImportFrom) and node.level==0:imports=[node.module or '']
                else:continue
                for imported in imports:
                    module=imported.split('.')[0]
                    if (root/(module+'.py')).is_file():pending.append(module)


if __name__=='__main__':unittest.main()
