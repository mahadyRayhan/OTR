# Tool Classification Reference: Health & Medical Science

## Clinical Data Warehouses and Common Data Models (OMOP, OHDSI, i2b2, REDCap)
- **Tools:** OMOP CDM, OHDSI ATLAS, OHDSI ACHILLES, i2b2, REDCap, TriNetX, PCORnet, Epic Clarity, Cerner HealtheIntent, FHIR, HL7 v2.
- **Compute Profile:** Database and query-bound. Not HPC. Sized by data volume and query concurrency, never by core count.
- **Hardware Recommendation:** Amazon Redshift, Aurora PostgreSQL, or RDS PostgreSQL. Athena over Parquet in S3 for ad-hoc cohort discovery.
- **PHI Exposure:** HIGH — these systems hold identified clinical records by design.
- **Compliance:** BAA required; RDS, Aurora, Redshift, Athena, and Glue are HIPAA eligible. Enforce row/column-level access control.
- **Source:** AWS HIPAA Eligible Services Reference. Tool grouping by system type `[VERIFY]`.

## Medical Imaging Toolkits (MONAI, nnU-Net, 3D Slicer, ITK, pydicom, Orthanc, XNAT)
- **Tools:** MONAI, nnU-Net, TotalSegmentator, 3D Slicer, ITK/SimpleITK, pydicom, dcm4che, Orthanc, XNAT, OHIF Viewer.
- **Compute Profile:** GPU-bound for training and inference (MONAI, nnU-Net, TotalSegmentator); light and single-node for viewers and DICOM servers (3D Slicer, Orthanc, XNAT).
- **Hardware Recommendation:** P-family for 3D training, G5 for 2D and inference. Orthanc/XNAT are persistent services on a small instance with S3 or EBS backing — not batch compute.
- **PHI Exposure:** HIGH. DICOM headers carry names, MRNs, and dates; pixel data may contain burned-in identifiers.
- **Compliance:** AWS HealthImaging, S3, FSx, SageMaker AI are eligible. SageMaker Ground Truth Plus and public labelling workforces are NOT.
- **Source:** AWS HIPAA Eligible Services Reference; 45 CFR §164.514(b)(2)(i). Tool grouping `[VERIFY]`.

## Clinical NLP and Automated De-identification (cTAKES, MedSpaCy, Philter, Comprehend Medical)
- **Tools:** Apache cTAKES, MedSpaCy, scispaCy, MetaMap, CLAMP, BioBERT, ClinicalBERT, GatorTron, Philter, MIST, Amazon Comprehend Medical.
- **Compute Profile:** Transformer models (BioBERT, ClinicalBERT, GatorTron) are GPU-bound; rule-based pipelines (cTAKES, MetaMap, Philter, regex) run adequately on CPU.
- **Hardware Recommendation:** G5 for transformer inference or fine-tuning; C-family for rule-based extraction. Comprehend Medical avoids self-hosting entirely and is HIPAA eligible.
- **PHI Exposure:** HIGHEST — identifiers are embedded in unstructured prose with no schema marking them.
- **Critical Rule:** Never send clinical free text to a service outside your BAA, including third-party LLM APIs. Automated de-identification tools reduce risk but do not by themselves establish Safe Harbor compliance.
- **Source:** AWS HIPAA Eligible Services Reference (Comprehend Medical). Tool grouping `[VERIFY]`.

## Survival Analysis and Biostatistics (lifelines, scikit-survival, Stan, R survival)
- **Tools:** lifelines, scikit-survival, R `survival`, R `survminer`, Stan / CmdStan, brms, JAGS, WinBUGS, statsmodels.
- **Compute Profile:** CPU-bound and modest. MCMC samplers (Stan, JAGS) parallelise across independent chains — typically 4–8 chains, so core counts beyond that give no benefit.
- **Hardware Recommendation:** A single C-family or R-family instance. Never an HPC cluster. Memory scales with dataset and posterior sample storage, not with cores.
- **PHI Exposure:** MODERATE — depends entirely on whether the analytic dataset retains dates and geography.
- **Spot Suitability:** EXCELLENT — short, restartable runs.
- **Source:** `[VERIFY]` Classified by algorithm structure (chain-parallel MCMC); confirm against each package's parallelisation documentation.

## Epidemic and Population Simulation (EpiModel, Covasim, EpiEstim)
- **Tools:** EpiModel, Covasim, GLEAM, EpiEstim, FluSight models, SEIR/agent-based frameworks.
- **Compute Profile:** Embarrassingly parallel across stochastic replicates and parameter sweeps. No inter-process communication.
- **Hardware Recommendation:** Many small C-family instances or AWS Batch. **EFA unnecessary.**
- **PHI Exposure:** USUALLY NONE — aggregate surveillance counts or synthetic populations.
- **Compliance:** Frequently requires **no BAA at all**. Do not apply the PHI control set reflexively to this class.
- **Spot Suitability:** EXCELLENT.
- **Source:** `[VERIFY]` Classified by workload structure (independent replicates), not a cited benchmark.

## Tabular Clinical Machine Learning (XGBoost, LightGBM, PyHealth, AutoGluon)
- **Tools:** XGBoost, LightGBM, CatBoost, scikit-learn, AutoGluon, TabNet, PyHealth, MLflow.
- **Compute Profile:** CPU-bound and modest; memory-bound on wide cohorts. GPU rarely justified for tabular data at clinical dataset sizes.
- **Hardware Recommendation:** A single memory-optimised R-family instance. SageMaker AI (excluding Studio Lab and public workforces) for managed training.
- **PHI Exposure:** HIGH if dates/ZIP/MRN retained; NONE once de-identified.
- **Common Error:** Provisioning GPU instances for gradient-boosted trees on a 100k-row cohort.
- **Spot Suitability:** GOOD.
- **Source:** AWS HIPAA Eligible Services Reference (SageMaker exclusions). Compute profile `[VERIFY]`.

## Clinical Genomics Pipelines (GATK, DRAGEN, HealthOmics)
- **Tools:** GATK, Illumina DRAGEN, BWA, samtools, VarSeq, Alissa Interpret, AWS HealthOmics.
- **Compute Profile:** Identical to research genomics — CPU-bound alignment, memory-bound variant calling. See `genomics_hpc_requirements.md`.
- **What Differs:** Governance, not compute. Genomic sequence is inherently re-identifiable and cannot be de-identified by field removal, so Safe Harbor fits poorly.
- **Hardware Recommendation:** C-family for alignment, R-family for variant calling, inside a BAA-covered account with KMS encryption throughout.
- **PHI Exposure:** HIGH and irreducible.
- **Source:** AWS HIPAA Eligible Services Reference (HealthOmics eligibility); `genomics_hpc_requirements.md`.

## Federated Learning Frameworks (NVIDIA FLARE, Flower, OpenFL)
- **Tools:** NVIDIA FLARE, Flower, OpenFL, Substra, TensorFlow Federated.
- **Compute Profile:** GPU-bound per site; the binding constraint is cross-site coordination and governance, not local throughput.
- **Hardware Recommendation:** G5/P-family per participating site; a small coordination service centrally that holds model weights only.
- **PHI Exposure:** HIGH locally; the design intent is NONE centrally.
- **Compliance:** Each site requires its own BAA and controls. `[VERIFY]` Whether weight or gradient sharing constitutes disclosure is a privacy-office question — model weights can leak training data under some attacks.
- **Spot Suitability:** POOR — synchronous rounds stall when a participant is reclaimed.
- **Source:** `[VERIFY]` Reasoned from federated architecture principles, not a retrieved AWS reference.

## Research Datasets and Access Regimes (MIMIC-IV, UK Biobank, All of Us, dbGaP)
- **Tools/Datasets:** MIMIC-IV, MIMIC-CXR, eICU, PhysioNet, UK Biobank, All of Us, dbGaP, TCGA, NIH STRIDES.
- **Why This Belongs Here:** Access regime, not compute, usually determines the architecture. Each dataset carries a Data Use Agreement with its own storage, egress, and sharing constraints — some prohibit copying data outside an approved environment entirely.
- **Compute Profile:** Varies by analysis; the datasets themselves impose governance, not hardware requirements.
- **PHI Exposure:** Typically de-identified but **credentialed** — de-identified status does not remove DUA obligations, and DUA terms frequently exceed HIPAA's technical requirements.
- **Architectural Rule:** Check the specific DUA before designing. A design that is HIPAA-sound can still breach a DUA.
- **`[VERIFY]`:** Per-dataset terms change and vary by approval; confirm current terms with the data custodian.
- **Source:** `[VERIFY]` Assembled from general knowledge of these access regimes; no primary source retrieved. Confirm against each dataset's current documentation.
