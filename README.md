# 🐍 Python Project with Virtual Environment Setup

This project uses a Python virtual environment (`venv`) to isolate dependencies.

---

## 🔧 Requirements


## 📦 Setting Up the Virtual Environment

### Linux(Bash)

```bash
# Create virtual environment in a folder named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Confirm activation
which python

```

### Windows (Powershell)

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create virtual environment in a folder named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Confirm activation
Get-Command python


Install Dependencies

pip install -r requirements.txt


Freeze Dependencies

pip freeze > requirements.txt






