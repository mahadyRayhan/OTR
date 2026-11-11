# sciweaver-sandbox: where model-written code runs (orchestrator/sandbox.py) and BioCoder answers are tested
# (journal_fixes/scripts/biocoder_eval.py). Runs get no network; database lookups go through orchestrator/lookups.py.
# Build: docker build -t sciweaver-sandbox:1 docker/sandbox
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 scikit-image==0.24.0 \
        scikit-learn==1.5.1 pysam==0.22.1 biopython==1.84 setuptools==72.1.0
# setuptools supplies distutils (LooseVersion), which Python 3.12 removed; 3.12 allows the nested
# f-string quotes that filled-in string placeholders produce.
