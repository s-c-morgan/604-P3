.PHONY: all venv clean clean-plot clean-img-data img-data notebook all-plot notebook-plot

# Abbreviations
PY := python3
PIP := python3 -m pip 
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/$(PY)
VENV_PIP := $(VENV_DIR)/bin/$(PIP)
DATA_DIR := data
PLOT_DIR := plots
SCR_DIR := scripts

# Default target first
all : img-data notebook-plot notebook

# Virtual environment
$(VENV_DIR) : requirements.txt # check requirement changes
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PIP) install -r requirements.txt
	touch $(VENV_DIR)

venv : $(VENV_DIR)

# Create data from images
$(DATA_DIR)/cilantro_stats.csv : $(VENV_DIR)
	$(VENV_PY) scripts/batch_cilantro_analyzer.py --dir $(DATA_DIR)/images
	touch $(DATA_DIR)/cilantro_stats.csv

img-data : $(DATA_DIR)/cilantro_stats.csv $(VENV_DIR)
	mv cilantro_stats.csv $(DATA_DIR)/cilantro_stats.csv

# Create notebook from report.py file
notebook : $(VENV_DIR)
	$(VENV_PY) -m jupytext --to ipynb project_notebook.py

# Generate all plots
all-plot : $(DATA_DIR)/cilantro_stats.csv $(VENV_DIR)
	mkdir -p $(PLOT_DIR)
	$(VENV_PY) plot_all_figures.py

# Generate all plots in the final notebooks
notebook-plot: $(DATA_DIR)/cilantro_stats.csv $(VENV_DIR)
	mkdir -p $(PLOT_DIR)
	$(VENV_PY) plot_final_notebook.py

# Clean ups
clean-img-data:
	rm $(DATA_DIR)/cilantro_stats.csv

clean-plot:
	rm -rf $(PLOT_DIR)

clean : 
	rm -rf $(VENV_DIR)
	rm -rf $(PLOT_DIR)
	rm -f $(DATA_DIR)/cilantro_stats.csv
	rm -f project_notebook.ipynb
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints"  -exec rm -rf {} +
