.PHONY: all venv clean clean-img-data img-data notebook all-plot notebook-plot

# Abbreviations
PY := python3
PIP := python3 -m pip 
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/$(PY)
VENV_PIP := $(VENV_DIR)/bin/$(PIP)
DATA_DIR := data
PLOT_DIR := plots
SCR_DIR := scripts
OUTPUT_DIR := output


# Default target first
all : notebook

# Virtual environment
$(VENV_DIR) : requirements.txt # check requirement changes
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PIP) install -r requirements.txt
	touch $(VENV_DIR)

venv : $(VENV_DIR)

# Create data from images
img-data : $(DATA_DIR)/cilantro_stats.scv
	$(VENV_PY) scripts/batch_cilantro_analyzer.py -dir $(DATA_DIR)/images

# Create notebook from report.py file
notebook : $(VENV_PY) -m jupytext --to ipynb project_notebook.py

# Generate all plots
all-plot : 
	$(VENV_PY) -m scripts/plot_eda.py
	$(VENV_PY) -m scripts/plot_group.series.py
	$(VENV_PY) -m scripts/test_fridge_layer_effects.py
	$(VENV_PY) -m scripts/test_bag_effect_for_infridge.py

# Generate all plots in the final notebooks
notebook-plot:
	$(VENV_PY) -m scripts/plot_final_notebook.py

# Clean ups
clean-img-data:
	rm $(DATA_DIR)/$(DATA_CSV)

clean : 
	rm -rf $(VENV_DIR)
	rm -rf $(PLOT_DIR)
	rm -rf $(OUTPUT_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
