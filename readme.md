# Ascomycete_T1-PKSs
From antismash database. https://antismash-db.secondarymetabolites.org/query
gene
BGC type = t1pks
Phylum = Ascomycota

# Clustalw/o alignment
/home/kkj/AI/clustalw/clustalw2 -infile=Ascomycete_T1-PKSs.fa -outfile=my_alignment.aln
/home/kkj/AI/clustalo/clustalo-1.2.4-Ubuntu-x86_6 -i filtered_Ascomycete_T1-PKSs.fa -o my_new_alignment.aln --outfmt=clustal -v --force