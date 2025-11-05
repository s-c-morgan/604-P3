.PHONY: all venv clean clean-img-data img-data report-notebook all-output report-output

# Abbreviations
PY := python3
PIP := python3 -m pip 
VENV_DIR := .venv
VENV_PY := $(VENV_DIR)/bin/$(PY)
VENV_PIP := $(VENV_DIR)/bin/$(PIP)
DATA_CSV := cilantro_stats.scv
DATA_DIR := data
PLOT_DIR := plots
SCR_DIR := scripts
OUTPUT_DIR := output


# Default target first
all : data report-output

# Virtual environment
$(VENV_DIR) : requirements.txt # check requirement changes
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PIP) install -r requirements.txt
	touch $(VENV_DIR)

venv : $(VENV_DIR)

# Create data from images
img-data : $(DATA_DIR)/$(DATA_CSV)

# Create notebook from .py file

# All-output

# Report-output

# Clean ups
clean-img-data:
	rm $(DATA_DIR)/$(DATA_CSV)

clean : 
	rm -rf $(VENV_DIR)
	rm -rf $(PLOT_DIR)
	rm -rf $(OUTPUT_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
