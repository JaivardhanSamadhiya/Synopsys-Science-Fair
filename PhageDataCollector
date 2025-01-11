from Bio import Entrez
import pandas as pd

# Set your email for Entrez
Entrez.email = "example@gmail.com"

def search_phages(query, max_records=100):
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records, retmode="xml")
    record = Entrez.read(handle)
    ids = record["IdList"]
    return ids

def fetch_phage_data(ids):
    phages = []
    for phage_id in ids:
        handle = Entrez.efetch(db="nucleotide", id=phage_id, retmode="xml")
        record = Entrez.read(handle)

        phage_name = record[0].get("GBSeq_locus", "Unknown")
        genome_size = record[0].get("GBSeq_length", "Unknown")
        host_range = record[0].get("GBSeq_organism", "Unknown")
        source_db = "NCBI GenBank"

        # Default to "Unknown"
        lytic_lysogenic = "Unknown"
        
        if "endolysin" in str(record[0]) or "holin" in str(record[0]):
            lytic_lysogenic = "Lytic"

        elif "integrase" in str(record[0]) or "prophage" in str(record[0]):
            lytic_lysogenic = "Lysogenic"
        
        phage_type = "Bacteriophage"
        
        phages.append([phage_name, genome_size, host_range, phage_type, source_db, lytic_lysogenic])
    
    return phages


phage_ids = search_phages("Staphylococcus phage", max_records=100)
phage_data = fetch_phage_data(phage_ids)


df = pd.DataFrame(phage_data, columns=[
    "Phage Name", "Genome Size (bp)", "Host Range", "Phage Type", "Source Database", "Lytic/Lysogenic"
])

# Save the data to a CSV file
df.to_csv("staph_phages_extended.csv", index=False)
