# Usage

This page explains how to use the features of the website.

## Create an Account

* Click "Sign Up" and follow the instructions to create an account.

    - Passwords are stored securely using the PBKDF2 algorithm with a SHA256 hash.

* If you already have an account, log in with your credentials.  

## Get Data Access

* Without data access you can still use the website; however, to actually view or download imaging files your account must be verified by one of the site administrators:

    - jiturner@bwh.harvard.edu
    - mding3@bwh.harvard.edu

They will grant your account access to view and download data.  

## Navigate Datasets

* Use the "File Explorer" page to navigate through all published datasets.

    - Published datasets are located at /data/nimlab/PUBLISHED_DATASETS/ on the server.
    - They are organized by author name and publication name. 
    - They generally follow the BIDS format (https://bids.neuroimaging.io/)
    - You can view files inside your browser.
    - You can also download files to your own machine.

* The "Search Datasets" page lets you search for datasets by tags or author names.

## Download CSVs  

* The "Download CSV" option appears on multiple pages. It downloads a CSV file containing the paths to imaging files within the dataset currently being viewed.

* This CSV can be used in Python notebooks to analyze the dataset. It contains paths to the imaging files in the dataset at /data/nimlab/PUBLISHED_DATASETS/, which is where the native imaging files are stored.

Please reach out to the site administrators if you have additional questions.