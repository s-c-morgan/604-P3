# STATS 604 Project 3 - The Cilantro experiment

In this project, we attempt to analyze various procedures for preserving the freshness of cilantro. Namely, we will attempt to measure if the following have any effect:

* Keeping cilantro in the fridge versus outside
* Keeping cilantro in a plastic bag versus not
* For refrigerated cilantro, keeping cilantro close to the cooling source at the top of the fridge versus farther away.

The data are saved in the this [Google Drive link](https://drive.google.com/drive/folders/1toR2fZkDzOGovN1LEGalaGmeIzfo1vmQ?usp=drive_link).

## Instruction

The Makefile is prepared with the following commands:

* `make all`, or `make`: generates csv from image data, plots notebook-relevant plots, and generate project's notebook.
* `make venv`: generate Python environment.
* `make clean`: clean up all generated files.
* `make clean-plot`: clean up plot directory.
* `make clean-img-data`: clean up generated csv file from image data.
* `make img-data`: processes images and generates csv from image data. Looks for image data in the directory `data/images`. One way to make sure this works is to download the `data` folder on the Google Drive link and save locally as directory `data`.
* `make notebook`: generates project's notebook.
* `make all-plot`: generates all plots that were potentially used for the analysis (thats a lot of plots).
* `make notebook-plot`: generates notebook-relevant plots
