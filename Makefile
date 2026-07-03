PYTHON ?= python
CONFIG ?= configs/baseline.yaml

.PHONY: install test run clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m src.load_data --config $(CONFIG)
	$(PYTHON) -m src.train --config $(CONFIG)
	$(PYTHON) -m src.evaluate --config $(CONFIG)
	$(PYTHON) -m src.report --config $(CONFIG)

clean:
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree('artifacts', ignore_errors=True); shutil.rmtree('.pytest_cache', ignore_errors=True); [p.unlink() for p in pathlib.Path('reports').glob('*.md') if p.is_file()]; [p.unlink() for p in pathlib.Path('reports/figures').glob('*') if p.is_file() and p.name != '.gitkeep']"
