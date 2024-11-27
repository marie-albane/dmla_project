.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y dmla || :
	@pip install -e .

run_preprocess:
	python -c 'from dmla.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from dmla.interface.main import train; train()'

run_pred:
	python -c 'from dmla.interface.main import pred; pred()'

run_evaluate:
	python -c 'from dmla.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

##################### RUN DMLA #####################

run_100_data:
	python -c 'from dmla.ml_logic.data import copier_coller; copier_coller(100,"training")'
	python -c 'from dmla.ml_logic.data import copier_coller; copier_coller(30,"validation")'
	python -c 'from dmla.ml_logic.data import copier_coller; copier_coller(20,"testing")'



##################### TESTS #####################

################### DATA SOURCES ACTIONS ################

show_sources_all:
	-ls -laR ~/.lewagon/mlops/data
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/mlops/data/
	mkdir ~/.lewagon/mlops/data/raw
	mkdir ~/.lewagon/mlops/data/processed
	mkdir ~/.lewagon/mlops/training_outputs
	mkdir ~/.lewagon/mlops/training_outputs/metrics
	mkdir ~/.lewagon/mlops/training_outputs/models
	mkdir ~/.lewagon/mlops/training_outputs/params

reset_local_files_with_csv_solutions: reset_local_files
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_all.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_all.csv


reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files


##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
