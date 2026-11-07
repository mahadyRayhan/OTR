# HPC Hardware Guidelines: Health & Medical Science

Provenance: author-written guidance without cited sources. Prefer the sourced files (aws_ec2_catalog_*.md, sourced_tool_requirements.md) when they disagree.

## Clinical Machine Learning on Structured EHR Data
- **Compute Profile:** CPU-bound and modest. Gradient-boosted trees (XGBoost, LightGBM) and regression on tabular cohorts of 10⁴–10⁷ rows rarely need a GPU or more than one node.
- **Hardware Recommendation:** A single memory-optimised R-family instance (r6i/r7i). Do not provision HPC or GPU capacity by default — the constraint is dataset size in memory, not FLOPs.
- **PHI Exposure:** HIGH if the cohort retains dates, ZIP codes, or MRNs; NONE once Safe Harbor de-identified.
- **Compliance Profile:** BAA required; SageMaker AI (excluding Studio Lab and public workforces) and RDS/Redshift are eligible. KMS at rest, TLS in transit, private subnet with VPC endpoints, CloudTrail data events on the cohort bucket.
- **Architectural Lever:** De-identify before the modelling stage where the analysis permits. This moves training out of the PHI boundary entirely.
- **Spot Suitability:** GOOD — training runs are short and restartable.
- **Source:** AWS HIPAA Eligible Services Reference; compute sizing `[VERIFY]` — reasoned from workload shape, not a cited benchmark.

## Medical Imaging Deep Learning (DICOM, MONAI, nnU-Net)
- **Compute Profile:** GPU-bound for training; large-object storage-bound for the dataset. 3D volumetric models (CT/MRI segmentation) are VRAM-limited more often than they are compute-limited.
- **Hardware Recommendation:** P-family (A100/H100 class) for 3D training; G5 (A10G, 24 GB) for 2D or inference. Stage the dataset on FSx for Lustre linked to S3 rather than pulling from S3 per epoch.
- **PHI Exposure:** HIGH and easily overlooked — DICOM headers carry patient name, MRN, and dates, and **pixel data itself may contain burned-in identifiers**. Full-face photographs and comparable images are themselves Safe Harbor identifiers.
- **Compliance Profile:** BAA required. AWS HealthImaging, S3, FSx, and SageMaker AI are eligible. **Critical trap:** SageMaker Ground Truth Plus and Public/Vendor labelling workforces are NOT HIPAA eligible — never route identifiable images to an external annotation workforce.
- **De-identification:** Strip DICOM headers and screen for burned-in text before the data leaves the PHI boundary.
- **Spot Suitability:** GOOD for training with checkpointing.
- **Source:** AWS HIPAA Eligible Services Reference (SageMaker exclusions, HealthImaging eligibility); 45 CFR §164.514(b)(2)(i) (facial images as identifiers).

## Clinical Natural Language Processing (free-text notes)
- **Compute Profile:** GPU-bound for transformer inference and fine-tuning; CPU-adequate for rule-based pipelines (cTAKES, regex extraction).
- **Hardware Recommendation:** G5 for transformer workloads; C-family for rule-based extraction. Amazon Comprehend Medical is HIPAA eligible and removes the need to self-host for entity extraction and PHI detection.
- **PHI Exposure:** HIGHEST OF ANY HEALTH WORKLOAD. Free-text notes contain unstructured identifiers — names, dates, addresses, phone numbers — embedded in prose, and no schema marks them. Structured de-identification does not apply.
- **Compliance Profile:** BAA mandatory. Treat every note as PHI until an automated de-identification pass plus human validation says otherwise. Comprehend Medical can detect PHI entities; it does not by itself constitute Safe Harbor de-identification.
- **Hard Rule:** Never send clinical free text to a non-HIPAA-eligible service. This includes third-party LLM APIs not covered by your BAA — a frequent and serious violation.
- **Spot Suitability:** GOOD for batch inference; MODERATE for fine-tuning.
- **Source:** AWS HIPAA Eligible Services Reference (Comprehend Medical eligibility); 45 CFR §164.514(b)(2)(i) (identifier categories).

## Clinical Genomic Sequencing and Genomic Medicine
- **Compute Profile:** Same compute shape as research genomics — CPU-bound alignment, memory-bound variant calling. See `genomics_hpc_requirements.md` for the hardware profile.
- **What Changes:** The compute is identical; the *governance* is not. Clinical genomic data is PHI, and genomic sequence is inherently re-identifiable — it cannot be meaningfully de-identified by removing fields, which makes Safe Harbor a poor fit.
- **Hardware Recommendation:** As per research genomics (C-family for alignment, R-family for variant calling), but within a BAA-covered account, private subnets, and KMS-encrypted storage throughout.
- **AWS Service:** HealthOmics is purpose-built for genomic storage and workflows and is HIPAA eligible. `[VERIFY]` — the eligibility is confirmed by the services reference, but the service's own feature detail could not be retrieved directly; check current documentation before designing around specific capabilities.
- **PHI Exposure:** HIGH and irreducible.
- **Spot Suitability:** GOOD for alignment (restartable), consistent with research genomics practice.
- **Source:** AWS HIPAA Eligible Services Reference (HealthOmics eligibility); `genomics_hpc_requirements.md` for compute profile.

## Epidemiological and Population Health Modelling
- **Compute Profile:** CPU-bound simulation, often embarrassingly parallel across parameter sets or stochastic replicates. Agent-based models (Covasim, EpiModel) and MCMC (Stan) parallelise across independent chains or runs.
- **Hardware Recommendation:** Many small/medium C-family instances, or Batch. EFA is unnecessary — replicates do not communicate.
- **PHI Exposure:** USUALLY NONE. This class typically uses aggregate counts, published surveillance data, or synthetic populations.
- **Compliance Profile:** **Often no BAA needed at all.** This is the important counterexample — applying the full PHI control set to public surveillance modelling adds cost and operational friction for no compliance benefit.
- **Diagnostic Question:** Does the model consume individual-level records, or aggregate counts? Aggregate-only work sits outside the PHI boundary.
- **Spot Suitability:** EXCELLENT — independent, restartable replicates.
- **Source:** `[VERIFY]` Compute profile reasoned from workload structure (independent replicates), not a cited benchmark. Compliance position follows from PHI definitions in 45 CFR §164.514.

## Multi-Site and Federated Learning
- **Compute Profile:** GPU-bound at each site; the distinguishing constraint is cross-site coordination, not local compute.
- **Why It Exists:** Federated designs are chosen precisely because PHI cannot leave each institution. The architecture is driven by data-governance boundaries, not performance.
- **Hardware Recommendation:** G5/P-family per site. Coordination via a central aggregation service holding only model weights — never records.
- **PHI Exposure:** HIGH within each site; the design goal is that it is NONE in transit and NONE centrally.
- **Compliance Profile:** Each site needs its own BAA and its own controls. The aggregator must be architected so that it *cannot* receive record-level data. `[VERIFY]` Model weights can leak training data under some attacks; whether gradient or weight sharing constitutes disclosure is a question for your privacy office, not an architectural given.
- **Spot Suitability:** POOR — synchronous federated rounds stall if a participant disappears.
- **Source:** `[VERIFY]` This section is reasoned from federated-learning architecture and PHI boundary principles, not drawn from a specific retrieved AWS reference.

## Clinical Registry and Cohort Analytics (OMOP, i2b2, REDCap)
- **Compute Profile:** Database and query-bound, not compute-bound. Workload is SQL over structured clinical warehouses, typically OMOP CDM or i2b2 star schemas.
- **Hardware Recommendation:** Amazon Redshift or Aurora PostgreSQL rather than an HPC cluster. Athena over Parquet in S3 for ad-hoc cohort discovery at lower cost.
- **PHI Exposure:** HIGH — clinical warehouses hold dates, MRNs, and geography by design; that is what makes them useful for cohort selection.
- **Compliance Profile:** BAA required. RDS (PostgreSQL), Aurora, Redshift, Athena, and Glue are all eligible. Enforce row- and column-level access control so analysts querying cohort counts never touch identifiers.
- **Recommended Pattern:** Two-tier — an identified tier for cohort selection with tight access control, and a de-identified extract tier where analysis actually happens.
- **Spot Suitability:** NOT APPLICABLE — persistent database services, not batch compute.
- **Source:** AWS HIPAA Eligible Services Reference (Redshift, Aurora, RDS, Athena, Glue eligibility). Two-tier pattern `[VERIFY]` — common practice, not a cited AWS reference architecture.
