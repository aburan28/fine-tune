import sys
from pathlib import Path

# Puts the repository root on sys.path so `import cryptorl` resolves whichever
# directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent))
