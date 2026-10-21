import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gai.core import doctor
print(json.dumps(doctor(),ensure_ascii=False,indent=2))
