# POC Supplier Entity Resolution

## Purpose
This project demonstrates a Proof of Concept for **entity resolution** of companies in a client’s database. The goal is to correctly identify and match suppliers, remove duplicates, and prepare a clean dataset for analysis.

## What the code does
1. Loads a file with companies and possible candidate matches.
2. Normalizes company names and removes stopwords.
3. Calculates matching scores based on:
   - company name
   - country
   - website
4. Selects the best candidate for each input row.
5. Marks each row as `matched` or `unmatched`.
6. Generates a summary report file with the results.

## How to run
Make sure Python and `pandas` are installed.
