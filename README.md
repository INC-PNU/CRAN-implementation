# CRAN-implementation
Cloud Radio Access Network (CRAN) Implementation for LoRa 

# Windows (CMD)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

# Windows (Git Bash)
```
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```
# Mac OS
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Initialize Database MongodB 

databases : cran

collections : 
- processed_iq_signals
- raw_iq_signals
