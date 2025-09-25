# POC Supplier Entity Resolution

## Purpose
This project demonstrates a Proof of Concept for **entity resolution** of companies in a client’s database. The goal is to correctly identify and match suppliers, remove duplicates, and prepare a clean dataset for analysis.

## What the code does
1. Loads a file with companies and possible candidate matches.
2. Normalizes company names and removes stopwords.
3. Calculates matching scores based on:
   - Name score (72%): receives the highest weight because, in general, the similarity of the company name is the strongest matching signal.
   - Country score (18%): has a lower weight, because companies can sometimes appear in multiple countries (subsidiaries, branches).
   - Website score (10%): has a small role, but helps with discrimination (a company with a website is an additional indicator of legitimacy and differentiation).
   - Threshold 0.55: means that a candidate is accepted as a match only if the final score is at least 55%.
4. Selects the best candidate for each input row.
5. Marks each row as `matched` or `unmatched`. This method detects duplicates (the same company written in different ways) and measures the correlation between the input and each candidate.
6. Generates a summary report file with the results.
7. OBS. The data in er_matches.xlsx are sorted by the 'final_score' column in descending order.

## How to run
Make sure PyCharm IDE and `pandas` package are installed. Use the PyCharm IDE to run the Python code.


