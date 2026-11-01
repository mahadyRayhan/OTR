# Compliance Baseline for Health Science Workloads on AWS

## Scope and Limitations of This Guidance
- **What This Is:** Architectural guidance for building health-research infrastructure on AWS. It describes technical controls and which AWS services are contractually permitted to hold PHI.
- **What This Is NOT:** Legal advice, a compliance certification, or a substitute for review by your institution's IRB, privacy office, or legal counsel. HIPAA compliance is a property of an organisation and its processes, not of an architecture diagram.
- **Mandatory Response Rule:** Any architecture that touches Protected Health Information (PHI) MUST state explicitly that the design requires review by the user's institutional privacy/compliance office before deployment. Never present a health architecture as "HIPAA compliant" — the correct phrasing is "built on HIPAA-eligible services with the following controls."
- **First Question To Ask:** Does this workload touch PHI at all? Much health research uses de-identified or public data (MIMIC, synthetic cohorts, published summary statistics), for which the entire compliance layer is unnecessary and adds pure cost.
- **Source:** AWS HIPAA Compliance page (aws.amazon.com/compliance/hipaa-compliance); general shared-responsibility framing.

## Business Associate Addendum (BAA) and HIPAA Eligible Services
- **The Contractual Gate:** AWS requires that a Covered Entity or Business Associate "agree not to use these HIPAA Eligible Services for any purpose or in any manner involving Protected Health Information without first entering into an AWS business associate agreement."
- **Practical Consequence:** A BAA must be in place BEFORE any PHI reaches AWS. This is an institutional procurement step, not something an engineer can configure. Architectures should call it out as a prerequisite.
- **Service Restriction:** Customers "should only process, store, and transmit protected health information (PHI) in the HIPAA-eligible services defined in the Business Associate Addendum." Using a non-eligible service for PHI violates the BAA regardless of how well it is encrypted.
- **Account Designation:** Customers designate accounts as HIPAA accounts and may use any service within them, but PHI handling remains restricted to eligible services.
- **Architectural Rule:** Before recommending any service for a PHI-touching workload, confirm it appears on the current HIPAA Eligible Services reference. The list changes; verify at design time.
- **Source:** AWS HIPAA Eligible Services Reference (aws.amazon.com/compliance/hipaa-eligible-services-reference).

## HIPAA Eligible Services by Category (verify currency at design time)
- **Compute:** Amazon EC2, AWS Lambda, Amazon ECS, Amazon EKS, AWS Fargate, AWS Batch.
- **Storage:** Amazon S3, Amazon EBS, Amazon EFS, Amazon FSx, AWS Storage Gateway. (FSx for Lustre is explicitly HIPAA eligible and assessed for ISO, PCI-DSS, and SOC.)
- **Analytics:** Amazon Athena, Amazon Redshift, AWS Glue, Amazon Kinesis (Streams, Firehose, Video).
- **Machine Learning:** Amazon SageMaker AI — *excluding* Studio Lab, Ground Truth Plus, and Public/Vendor Workforce — plus Amazon Comprehend Medical, Amazon Rekognition, Amazon Textract.
- **Healthcare-Specific:** AWS HealthLake, AWS HealthOmics, AWS HealthImaging, Amazon Comprehend Medical.
- **Databases:** Amazon RDS (SQL Server, MySQL, Oracle, PostgreSQL, Db2, MariaDB), Amazon Aurora, Amazon DynamoDB.
- **Common Trap:** SageMaker's exclusions matter. Ground Truth Plus and public labelling workforces are NOT eligible — routing PHI to a human labelling workforce is a frequent and serious design error in medical imaging annotation pipelines.
- **Source:** AWS HIPAA Eligible Services Reference; AWS FSx for Lustre User Guide (FSx HIPAA eligibility).

## De-identification: Safe Harbor vs Expert Determination
- **Why It Matters Architecturally:** De-identified data is not PHI. Successfully de-identifying upstream removes the entire BAA/eligible-service constraint from everything downstream — usually the single highest-leverage architectural decision in a health project.
- **Two Methods (45 CFR §164.514(b)):** *Safe Harbor* is rule-based — remove the enumerated identifier categories and retain no actual knowledge that the remainder could identify someone. *Expert Determination* is risk-based — a qualified expert documents that re-identification risk is "very small," which permits retaining richer data such as month-level dates or regional geography, at the cost of expert analysis and ongoing governance.
- **Choosing:** Safe Harbor is cheaper and mechanical but destroys temporal granularity, often ruining longitudinal analyses. Expert Determination preserves analytic utility but requires a qualified expert and documented risk management.
- **Architectural Pattern:** Perform de-identification inside the PHI boundary, then export the de-identified derivative to a separate account or VPC with a relaxed control set. Do not run the whole pipeline under maximum controls when only the ingest stage needs them.
- **Source:** 45 CFR §164.514(b)(2)(i); HHS de-identification guidance (accessed via secondary summaries — the HHS page itself blocked automated retrieval).

## Safe Harbor Identifier Categories (must be removed)
- **Direct Identifiers:** Names (individual, relatives, employers, household members); telephone numbers; fax numbers; email addresses; Social Security numbers; medical record numbers; health plan beneficiary numbers; account numbers; certificate and licence numbers.
- **Technical Identifiers:** Vehicle identifiers and serial numbers including licence plates; device identifiers and serial numbers; web URLs; IP addresses; biometric identifiers including fingerprints and voice prints; full-face photographs and comparable images.
- **Geography:** All geographic subdivisions smaller than a state. The first three ZIP digits may be retained only if the Census-defined area sharing those digits exceeds 20,000 people; otherwise it must be changed to 000.
- **Dates:** All date elements except year — birth date, admission date, discharge date, date of death. All ages over 89 (and any date implying such an age) must be aggregated into a single "90 or older" category.
- **Catch-All:** Any other unique identifying number, characteristic, or code — including rare diagnoses combined with small geography, or unusual occupations functioning as de facto identifiers.
- **`[VERIFY]`:** Sources enumerate these categories as 18 items but split them inconsistently (dates vs ages over 89 are sometimes one category, sometimes two). The categories above are complete; confirm the exact enumeration and its application with your privacy office rather than relying on this count.
- **Source:** 45 CFR §164.514(b)(2)(i), via secondary summaries of HHS guidance.

## Baseline Technical Controls for PHI Workloads
- **Encryption at Rest:** KMS-managed keys on every store holding PHI — S3, EBS, FSx, RDS. FSx for Lustre encrypts at rest automatically using KMS. Prefer customer-managed keys (CMKs) over AWS-managed keys where key rotation and access must be auditable.
- **Encryption in Transit:** TLS for all service endpoints. FSx for Lustre encrypts in transit automatically in certain Regions when accessed from supported EC2 instance types — confirm both Region and instance type rather than assuming.
- **Network Isolation:** PHI workloads belong in private subnets with no direct internet gateway; reach AWS services through VPC endpoints so traffic does not traverse the public internet.
- **Access Control:** IAM least-privilege, scoped to the specific buckets, prefixes, and file systems holding PHI. Avoid account-wide wildcards.
- **Audit:** CloudTrail enabled with log-file validation, plus S3 server access logging or CloudTrail data events on PHI buckets. Retain per your institution's policy — audit trails are typically the first thing requested in a breach investigation.
- **`[VERIFY]`:** Specific retention periods and key-rotation intervals are institution- and state-dependent. This section lists control *types*; your compliance office sets the parameters.
- **Source:** AWS FSx for Lustre User Guide (encryption behaviour); AWS HIPAA Compliance page. Control-type list assembled from AWS security guidance rather than a single cited page.
