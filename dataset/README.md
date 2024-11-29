# CIC-IDS2017 Dataset - README

## Overview
The CIC-IDS2017 dataset is a benchmark dataset for Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS). It contains realistic network traffic, including both benign activity and a variety of common network attacks. The dataset was collected over five days, from July 3 to July 7, 2017, and reflects real-world behavior by simulating interactions of 25 users with different protocols such as HTTP, HTTPS, FTP, SSH, and email.

This dataset includes PCAP files (full packet captures) as well as CSV files containing network flow features extracted using CICFlowMeter. The data is labeled with the specific attack types and covers a wide range of attacks, such as Brute Force, DoS, DDoS, Heartbleed, Web Attacks, Infiltration, and Botnets. It is commonly used for machine learning and deep learning research for cybersecurity.

## Folder Structure
The dataset folder contains the following files and directories:

- **PCAP Files**: Full packet captures of the network traffic.
  - `Day1.pcap`: Monday - Only benign traffic.
  - `Day2.pcap`: Tuesday - Benign traffic + Brute Force (FTP, SSH) attacks.
  - `Day3.pcap`: Wednesday - Benign traffic + DoS, Heartbleed attacks.
  - `Day4.pcap`: Thursday - Benign traffic + Web, Infiltration attacks.
  - `Day5.pcap`: Friday - Benign traffic + Botnet, DDoS, Port Scan attacks.

- **CSV Files**: Network flow data extracted from the PCAP files using CICFlowMeter, containing over 80 features for machine learning purposes.
  - `MachineLearningCSV.zip`: Contains CSV files with labeled network flows for each day.
  
- **GeneratedLabelledFlows.zip**: Contains labeled network flows with metadata for each day.

- **Documentation**: Includes details about the dataset, attack scenarios, and network setup.
  - `CICIDS2017_Documentation.pdf`: Explains the dataset structure, attack details, and network configuration.

## Dataset Details
- **Monday, July 3, 2017**: Contains only benign traffic (normal user activity).
- **Tuesday, July 4, 2017**: Includes Brute Force attacks (FTP and SSH) along with normal traffic.
- **Wednesday, July 5, 2017**: Contains DoS (Slowloris, Hulk, Slowhttptest, GoldenEye) and Heartbleed attacks.
- **Thursday, July 6, 2017**: Includes Web attacks (Brute Force, XSS, SQL Injection) and Infiltration attacks.
- **Friday, July 7, 2017**: Botnet, Port Scans, and DDoS attacks are executed.

## Attack Types
The dataset includes the following types of attacks:
- **Brute Force** (FTP, SSH)
- **DoS** (Slowloris, Hulk, GoldenEye, etc.)
- **Heartbleed**
- **Web Attacks** (Brute Force, XSS, SQL Injection)
- **Infiltration**
- **Botnet (ARES)**
- **DDoS (LOIT)**
- **Port Scans**

## Usage
This dataset is widely used for:
- Intrusion detection research.
- Machine learning and deep learning model training for cybersecurity.
- Network traffic analysis.

### How to Use:
1. **PCAP Analysis**: Use network analysis tools like Wireshark to inspect PCAP files.
2. **CSV Files**: Load CSV files into machine learning frameworks (such as Python's pandas library) for training or testing intrusion detection models.
3. **Feature Engineering**: Use the extracted features in the CSV files for developing and testing anomaly detection models.

## Citation
If you are using the CICIDS2017 dataset in your research, please cite the following paper:

**Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani**, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

## License
This dataset is publicly available for research purposes. Please refer to the dataset documentation for further details on the terms of use.

## Contact
For more information or inquiries, please visit the [Canadian Institute for Cybersecurity (CIC)](https://www.unb.ca/cic/datasets/ids-2017.html).
