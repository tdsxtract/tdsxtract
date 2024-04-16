
SHELL := /bin/bash

.PHONY: clean lint req doc test

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = tdsxtract
PYTHON_INTERPRETER = python3
HOSTING = gitlab
VERSION=$(shell python3 -c "import tdsxtract; print(tdsxtract.__version__)")

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################


default:
	@echo "\"make save\"?"


## Set up python interpreter environment
env:
ifeq (True,$(HAS_CONDA))
		@echo -e ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo -e ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo -e ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo -e ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

version:
	@echo $(VERSION)


## Test python environment is setup correctly
testenv:
	source activate $(PROJECT_NAME); \
	$(PYTHON_INTERPRETER) .ci/testenv.py


## Install Python dependencies
req: testenv
	source activate $(PROJECT_NAME)
	# $(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	.ci/installreq requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e.

## Install Python dependencies for dev and test
dev:
	$(PYTHON_INTERPRETER) -m pip install -e .[dev]
	

install:
	pip install -e .


## Delete generated files
clean:
	@find . | grep -E "(*.pvd*.xdmf|*.msh|*.pvtu|*.vtu|*.pvd|jitfailure*|tmp|__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf .pytest_cache $(PROJECT_NAME).egg-info/ build/ dist/ tmp/ htmlcov/
	cd doc && make clean
	rm -rf .coverage htmlcov coverage.xml


## Lint using flake8
lint:
	flake8 --exit-zero setup.py $(PROJECT_NAME)/ test/*.py examples/

## Check for duplicated code
dup:
	pylint --exit-zero -f colorized --disable=all --enable=similarities $(PROJECT_NAME)

## Check for missing docstring
dcstr:
	pydocstyle ./$(PROJECT_NAME)  || true

## Metric for complexity
rad:
	radon cc ./$(PROJECT_NAME) -a -nb

## Run all code checks
lint-all: lint dup dcstr rad

## Reformat code
style:
	@echo "Styling..."
	isort .
	black .

## Push to gitlab
gl:
	@echo "Pushing to gitlab..."
	git add -A
	@read -p "Enter commit message: " MSG; \
	git commit -a -m "$$MSG"
	git push origin main

## Clean, reformat and push to gitlab
save: clean style gl


## Install requirements for building doc
doc-req:
	cd doc && pip install -r requirements.txt


## Build html doc
doc:
	cd doc && make dochtml

# 
# 
# ## Build html doc (without examples)
# doc-noplot:
# 	cd doc && make clean && make html-noplot && make postpro-html




## Run the test suite
test:
	rm -rf .coverage htmlcov
	pytest ./test --cov=./$(PROJECT_NAME) --cov-report term --cov-report html --cov-report xml  
	

## Tag and push tags
tag:
	# Make sure we're on the main branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags


## Package 
package: setup.py
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	rm -f dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel --universal

## Upload to pipy
pipy: package
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	twine upload dist/*


## Tag and upload to pipy
publish: tag pipy


###############





#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo -e "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo -e
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
