# Cloud Pricing and Optimization for Bioinformatics

## EC2 Spot Instances
- Bioinformatics pipelines (like Nextflow and Snakemake) are generally fault-tolerant and stateless. This makes them perfect candidates for AWS Spot Instances or GCP Preemptible VMs.
- **Cost Savings:** Using Spot Instances reduces compute costs by up to 90%. 
- **Drawback:** Instances can be terminated with 2 minutes' notice. Nextflow handles this by automatically resubmitting the failed job.

## Storage Tiering
- **Active Processing:** Use AWS S3 Standard or EFS for active pipeline runs.
- **Archival:** Once FASTQ and BAM files are processed, use S3 Lifecycle Policies to move them to S3 Glacier Deep Archive within 30 days to save ~95% on storage costs.