#!/bin/bash

conda init
conda activate transformers

# Place file inside transformers repo.

python utils/check_docstrings.py --check_all
python utils/check_doc_toc.py
python utils/sort_auto_mappings.py
python utils/custom_init_isort.py
ruff format examples tests src utils
ruff check examples tests src utils
python utils/check_copies.py
python utils/check_modular_conversion.py
python utils/check_dummies.py
python utils/check_repo.py
python utils/check_inits.py
python utils/check_config_docstrings.py
python utils/check_config_attributes.py
python utils/check_doctest_list.py
make deps_table_check_updated
python utils/update_metadata.py
python utils/check_docstrings.py
