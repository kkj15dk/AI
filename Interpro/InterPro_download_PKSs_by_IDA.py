#!/usr/bin/env python3

# standard library modules
import sys, errno, re, json, ssl, os
from urllib import request
from urllib.error import HTTPError
from time import sleep

# Script to download all PKSs with a specific InterPro domain architecture (IDA) (with KS, AT and ACP. Without Adenylation domain, ABC transporter, choline acyltransferase, aminotransferase, PKS docking domain, NMO, or condensation domains) from InterPro, in taxonomic group 4751 (Fungi) # 4890 (Ascomycetes)

IDA_API_URL = "https://www.ebi.ac.uk/interpro/api/entry/?ida_search=IPR014031%2CIPR014030%2CIPR014043%2CIPR009081&ida_ignore=IPR000873%2CPF00005%2CIPR039551%2CIPR004839%2CIPR015083%2CIPR004136%2CIPR001242"

BASE_URL = "https://www.ebi.ac.uk/interpro/wwwapi/protein/UniProt/taxonomy/uniprot/4751/?ida=" # 4890/?ida= (for ascomycetes)

URL_EXTENSION = "&page_size=200&extra_fields=sequence%2Cgo_terms"

HEADER_SEPARATOR = "|"
LINE_LENGTH = 80

IDA_FILE = "new4_IDA_IDs.fa"
OUTPUT_FILE = "new4_PKSs.fa"

def get_ida_ids():
  output_handle = open(IDA_FILE, "w")
  #disable SSL verification to avoid config issues
  context = ssl._create_unverified_context()

  next = IDA_API_URL
  last_page = False

  
  attempts = 0
  while next:
    try:
      req = request.Request(next, headers={"Accept": "application/json"})
      res = request.urlopen(req, context=context)
      # If the API times out due a long running query
      if res.status == 408:
        # wait just over a minute
        sleep(61)
        # then continue this loop with the same URL
        continue
      elif res.status == 204:
        #no data so leave loop
        break
      payload = json.loads(res.read().decode())
      next = payload["next"]
      attempts = 0
      if not next:
        last_page = True
    except HTTPError as e:
      if e.code == 408:
        sleep(61)
        continue
      else:
        # If there is a different HTTP error, it will re-try 3 times before failing
        if attempts < 3:
          attempts += 1
          sleep(61)
          continue
        else:
          sys.stderr.write("LAST URL: " + next)
          raise e

    for i, item in enumerate(payload["results"]):
      
      # Don't include entries with multiple KS or AR domains (IPR014031, Ketoacyl_synth_C) (IPR014043, At doamin)
      KSdomains = [domain for domain in item["representative"]["domains"] if domain["accession"] == "IPR014031"]
      if len(KSdomains) > 1:
        print(str(len(KSdomains)) + " KS domains found in IDA: " + item["ida_id"] + " skipping")
        continue
      ATdomains = [domain for domain in item["representative"]["domains"] if domain["accession"] == "IPR014043"]
      if len(ATdomains) > 1:
        print(str(len(ATdomains)) + " AT domains found in IDA: " + item["ida_id"] + " skipping")
        continue

      if ("ida_id" in item):
        ida_id = item["ida_id"]
      else:
        ida_id = "N/A"
      output_handle.write(ida_id + "\n")
      
    # Don't overload the server, give it time before asking for more
    if next:
      sleep(1)

def output_list(base_url):
  output_handle = open(OUTPUT_FILE, "a")
  #disable SSL verification to avoid config issues
  context = ssl._create_unverified_context()

  next = base_url
  last_page = False

  
  attempts = 0
  while next:
    try:
      req = request.Request(next, headers={"Accept": "application/json"})
      res = request.urlopen(req, context=context)
      # If the API times out due a long running query
      if res.status == 408:
        # wait just over a minute
        sleep(61)
        # then continue this loop with the same URL
        continue
      elif res.status == 204:
        #no data so leave loop
        break
      payload = json.loads(res.read().decode())
      next = payload["next"]
      attempts = 0
      if not next:
        last_page = True
    except HTTPError as e:
      if e.code == 408:
        sleep(61)
        continue
      else:
        # If there is a different HTTP error, it wil re-try 3 times before failing
        if attempts < 3:
          attempts += 1
          sleep(61)
          continue
        else:
          sys.stderr.write("LAST URL: " + next)
          raise e

    for i, item in enumerate(payload["results"]):
      entries = None
      if ("entry_subset" in item):
        print("entry_subset!")
        entries = item["entry_subset"]
      elif ("entries" in item):
        print("entries!")
        entries = item["entries"]
      
      # if entries is not None:
      #   entries_header = "-".join(
      #     [entry["accession"] + "(" + ";".join(
      #       [
      #         ",".join(
      #           [ str(fragment["start"]) + "..." + str(fragment["end"]) 
      #             for fragment in locations["fragments"]]
      #         ) for locations in entry["entry_protein_locations"]
      #       ]
      #     ) + ")" for entry in entries]
      #   )
      #   output_handle.write(">" + item["metadata"]["accession"] + HEADER_SEPARATOR
      #                     + entries_header + HEADER_SEPARATOR
      #                     + item["metadata"]["name"] + "\n")
      # else:
      go_terms = [term for term in item["extra_fields"]["go_terms"] if term["identifier"] == "GO:0022857" or term["identifier"] == "GO:0055085"]
      if go_terms:
        print("Unwanted GO term found! in entry: " + item["metadata"]["accession"])
        print([go_term["identifier"] for go_term in go_terms])
        continue
      output_handle.write(">" + item["metadata"]["accession"] + HEADER_SEPARATOR + item["metadata"]["name"] + "\n")

      seq = item["extra_fields"]["sequence"]
      fastaSeqFragments = [seq[0+i:LINE_LENGTH+i] for i in range(0, len(seq), LINE_LENGTH)]
      for fastaSeqFragment in fastaSeqFragments:
        output_handle.write(fastaSeqFragment + "\n")
      
    # Don't overload the server, give it time before asking for more
    if next:
      sleep(1)

if __name__ == "__main__":
  if os.path.isfile(IDA_FILE):
    print("Using IDA IDs from file: " + IDA_FILE)
  else:
    get_ida_ids()
  if os.path.isfile(OUTPUT_FILE):
    print(OUTPUT_FILE + " already exists, please remove or rename it before running this script")
  else:
    open(OUTPUT_FILE, 'a').close()
  f = open(IDA_FILE, 'r')
  ida_ids = [id.rstrip('\n') for id in f.readlines()]
  for id in ida_ids:
    if id == "N/A":
      print("Missing id on line " + str(ida_ids.index(id)) + " continuing")
      continue
    base_url = BASE_URL + id + URL_EXTENSION
    output_list(base_url)
