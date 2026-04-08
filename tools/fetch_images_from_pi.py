"""
Bilder vom Pi auf den PC übertragen (via SFTP/SSH).

Verwendung:
    python fetch_images_from_pi.py

Voraussetzung:
    pip install paramiko
"""
import paramiko
from pathlib import Path

# --- Verbindungseinstellungen ------------------------------------------------
PI_HOST     = "192.168.178.177"
PI_USER     = "ir"
PI_PASSWORD = "pir"
PI_IMG_DIR  = "/home/ir/industrielle-robotik-wise25-balancing-robot/tests/output"
# -----------------------------------------------------------------------------

LOCAL_OUT_DIR = Path(__file__).resolve().parent / "images_from_pi"
LOCAL_OUT_DIR.mkdir(exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(PI_HOST, username=PI_USER, password=PI_PASSWORD)

sftp = ssh.open_sftp()

try:
    files = sftp.listdir(PI_IMG_DIR)
except FileNotFoundError:
    print(f"Ordner nicht gefunden auf dem Pi: {PI_IMG_DIR}")
    sftp.close()
    ssh.close()
    raise SystemExit(1)

if not files:
    print("Keine Dateien gefunden.")
else:
    for filename in files:
        remote_path = f"{PI_IMG_DIR}/{filename}"
        local_path  = LOCAL_OUT_DIR / filename
        sftp.get(remote_path, str(local_path))
        print(f"Übertragen: {filename} -> {local_path}")

sftp.close()
ssh.close()

print(f"\nFertig. {len(files)} Datei(en) in: {LOCAL_OUT_DIR}")
