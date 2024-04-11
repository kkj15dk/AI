# Ascomycete_T1-PKSs
From antismash database. https://antismash-db.secondarymetabolites.org/query
gene
BGC type = t1pks
Phylum = Ascomycota

# protein-matching-IPR020841
https://www.ebi.ac.uk/interpro/entry/InterPro/IPR020841/protein/UniProt/taxonomy/uniprot/4890/#table

# Clustalw/o alignment
/home/kkj/AI/clustalw/clustalw2 -infile=filtered_PKSs.fa -outfile=my_alignment.aln
/home/kkj/AI/clustalo/clustalo-1.2.4-Ubuntu-x86_64 -i filtered_PKSs.fa -o clustalo_alignment.aln --outfmt=fasta -v --threads=16

with screen:
screen -L -Logfile logs/screen_log.txt -dmS clustalo /home/kkj/AI/clustalo/clustalo-1.2.4-Ubuntu-x86_64 -i filtered_PKSs.fa -o clustalo_alignment.aln --outfmt=fasta -v --threads=16 > clustalo_stdout.out