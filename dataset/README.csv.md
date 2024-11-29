
### **How This Script Works**
1. **Base URL**: Points to the directory with `.zip` files.
2. **BeautifulSoup**: Parses the HTML to find all file links ending with `.zip`.
3. **Download Logic**:
   - Each `.zip` file is downloaded and saved in the `CICIDS2017_CSVs` folder.
4. **Output Directory**: Ensures a dedicated folder for downloads.

---

### **Steps to Run the Script**
1. **Install Required Libraries**:
   - If not already installed, run:
     ```bash
     pip install requests beautifulsoup4
     ```
2. **Save and Execute**:
   - Save the script to a file, e.g., `download_csvs.py`.
   - Run the script:
     ```bash
     python request_csvs.py
     ```

---

### **Next Steps After Download**
1. **Extract the ZIP Files**:
   ```python
   import zipfile
   import glob

   # Extract all ZIP files in the output directory
   zip_files = glob.glob("CICIDS2017_CSVs/*.zip")
   for zip_file in zip_files:
       with zipfile.ZipFile(zip_file, 'r') as zip_ref:
           zip_ref.extractall("CICIDS2017_CSVs")
           print(f"Extracted: {zip_file}")
   ```

2. **Verify Data Integrity**:
   Use the provided `.md5` files to ensure the downloaded files are not corrupted:
   ```bash
   md5sum -c GeneratedLabelledFlows.md5
   md5sum -c MachineLearningCSV.md5
   ```