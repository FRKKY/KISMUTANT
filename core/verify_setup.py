#!/usr/bin/env python3
"""
SETUP VERIFICATION SCRIPT

Run this script to verify your setup is correct before starting the system.
It checks:
1. Python version
2. Required packages
3. Configuration files
4. API credentials
5. Database connectivity
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Verify Python version."""
    print("Checking Python version...", end=" ")
    
    required = (3, 9)
    current = sys.version_info[:2]
    
    if current >= required:
        print(f"✓ Python {current[0]}.{current[1]}")
        return True
    else:
        print(f"✗ Python {current[0]}.{current[1]} (requires {required[0]}.{required[1]}+)")
        return False


def check_packages():
    """Verify required packages are installed."""
    print("\nChecking required packages...")
    
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("httpx", "httpx"),
        ("yaml", "pyyaml"),
        ("sqlalchemy", "sqlalchemy"),
        ("loguru", "loguru"),
        ("pydantic", "pydantic"),
    ]
    
    all_ok = True
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - run: pip install {package_name}")
            all_ok = False
    
    return all_ok


def check_config_files():
    """Verify configuration files exist."""
    print("\nChecking configuration files...")
    
    config_dir = Path(__file__).parent.parent / "config"
    
    required_files = [
        ("settings.yaml", True),
        ("credentials.yaml", False),  # May not exist yet
    ]
    
    all_ok = True
    for filename, required in required_files:
        filepath = config_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        elif required:
            print(f"  ✗ {filename} - MISSING (required)")
            all_ok = False
        else:
            print(f"  ⚠ {filename} - not found (copy from {filename.replace('.yaml', '.example.yaml')})")
    
    return all_ok


def check_credentials():
    """Verify API credentials are configured."""
    print("\nChecking API credentials...")
    
    credentials_path = Path(__file__).parent.parent / "config" / "credentials.yaml"
    
    if not credentials_path.exists():
        print("  ⚠ credentials.yaml not found")
        print("    1. Copy credentials.example.yaml to credentials.yaml")
        print("    2. Fill in your KIS API credentials")
        return False
    
    import yaml
    
    try:
        with open(credentials_path, 'r') as f:
            creds = yaml.safe_load(f)
        
        kis = creds.get("kis", {})
        paper = kis.get("paper", {})
        
        # Check for placeholder values
        if "YOUR_" in str(paper.get("app_key", "")):
            print("  ⚠ Paper trading credentials not configured")
            print("    Edit config/credentials.yaml with your KIS API keys")
            return False
        
        if paper.get("app_key") and paper.get("app_secret"):
            print("  ✓ Paper trading credentials configured")
            return True
        else:
            print("  ⚠ Paper trading credentials incomplete")
            return False
            
    except Exception as e:
        print(f"  ✗ Error reading credentials: {e}")
        return False


def check_directories():
    """Verify required directories exist."""
    print("\nChecking directories...")
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "config",
        "memory",
        "logs",
        "core",
        "perception",
        "hypothesis",
        "validation",
        "portfolio",
        "execution",
    ]
    
    all_ok = True
    for dirname in required_dirs:
        dirpath = project_root / dirname
        if dirpath.exists():
            print(f"  ✓ {dirname}/")
        else:
            print(f"  ✗ {dirname}/ - MISSING")
            all_ok = False
    
    return all_ok


def check_database():
    """Verify database can be initialized."""
    print("\nChecking database...")
    
    try:
        from memory.models import Database
        
        # Use in-memory database for test
        db = Database("sqlite:///:memory:")
        db.create_tables()
        
        print("  ✓ Database schema valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def check_invariants():
    """Verify invariants module loads correctly."""
    print("\nChecking invariants...")
    
    try:
        from core.invariants import INVARIANTS
        
        print(f"  ✓ Max single position: {float(INVARIANTS.MAX_SINGLE_POSITION_PCT)*100}%")
        print(f"  ✓ Drawdown kill switch: {float(INVARIANTS.DRAWDOWN_KILL_PCT)*100}%")
        print(f"  ✓ Max leverage: {float(INVARIANTS.MAX_LEVERAGE)}x")
        return True
        
    except Exception as e:
        print(f"  ✗ Invariants error: {e}")
        return False


def check_api_connection():
    """Test API connection (optional)."""
    print("\nChecking API connection...")
    
    credentials_path = Path(__file__).parent.parent / "config" / "credentials.yaml"
    
    if not credentials_path.exists():
        print("  ⚠ Skipped - no credentials file")
        return None
    
    try:
        from execution.broker import KISBroker
        
        broker = KISBroker(mode="paper")
        
        if broker.test_connection():
            print("  ✓ API connection successful")
            broker.close()
            return True
        else:
            print("  ✗ API connection failed")
            broker.close()
            return False
            
    except Exception as e:
        print(f"  ⚠ API test skipped: {e}")
        return None


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("LIVING TRADING SYSTEM - SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Required Packages", check_packages()))
    results.append(("Directories", check_directories()))
    results.append(("Config Files", check_config_files()))
    results.append(("Credentials", check_credentials()))
    results.append(("Database", check_database()))
    results.append(("Invariants", check_invariants()))
    
    # Optional: API connection test
    api_result = check_api_connection()
    if api_result is not None:
        results.append(("API Connection", api_result))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All checks passed! System is ready to run.")
        print("\nTo start the system:")
        print("  python main.py --mode paper")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
