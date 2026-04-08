# industrielle-robotik-wise25-balancing-robot

---

## Tutorial 1: Code auf GitHub pushen (vom PC)

### Voraussetzungen
- [Git](https://git-scm.com/downloads) ist installiert
- Du hast einen GitHub-Account und bist dem Repository hinzugefügt worden

### Schritte

**1. Repository einmalig klonen (nur beim ersten Mal)**
```bash
git clone https://github.com/cyHeff5/industrielle-robotik-wise25-balancing-robot.git
cd industrielle-robotik-wise25-balancing-robot
```

**2. Aktuellen Stand vom Remote holen (vor jeder Arbeitssession)**
```bash
git pull
```

**3. Änderungen vorbereiten und committen**
```bash
# Alle geänderten Dateien zum Commit hinzufügen
git add .

# Oder gezielt einzelne Dateien:
git add pfad/zur/datei.py

# Commit mit Nachricht erstellen
git commit -m "Kurze Beschreibung der Änderung"
```

**4. Code auf GitHub hochladen**
```bash
git push
```

> Beim ersten Push wird Git nach deinen GitHub-Zugangsdaten fragen.
> Empfehlung: [SSH-Key einrichten](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) oder einen GitHub Personal Access Token verwenden.

[Tutorial video](https://www.youtube.com/watch?v=snCP3c7wXw0)

---

## Tutorial 2: Code vom GitHub auf den Pi laden

Der Pi ist über den Hotspot von Hanan erreichbar. Je nachdem, ob die IP-Adresse bekannt ist, gibt es zwei Wege.

---

### Option A: Verbindung per PuTTY (wenn IP bekannt)

**Voraussetzungen**
- [PuTTY](https://www.putty.org/) ist installiert
- PC und Pi sind beide mit Hanans Hotspot verbunden
- Die IP-Adresse des Pi ist bekannt (z.B. `192.168.x.x`)

**Schritte**

1. PuTTY öffnen
2. Im Feld **Host Name (or IP address)** die IP des Pi eintragen
3. **Port**: `22`, **Connection type**: `SSH`
4. Auf **Open** klicken
5. Login mit Pi-Username und Passwort

**IP-Adresse des Pi herausfinden:**
Falls der Pi an einen Bildschirm angeschlossen ist oder war, kann man direkt am Pi-Terminal eingeben:
```bash
ifconfig
```

**Code auf den Pi holen:**

```bash
cd industrielle-robotik-wise25-balancing-robot
git pull
```

---

### Option B: Pi direkt per Bildschirm und Tastatur bedienen

Falls keine Netzwerkverbindung verfügbar ist oder die IP nicht bekannt ist, einfach per Bildschirm verbinden. Um vom Github Repo zu pullen, muss der Pi aber mit dem Internet verbunden sein.

**Schritte**

1. Monitor per HDMI und Tastatur an den Pi anschließen
2. Pi einschalten (oder neu starten)
3. Repository aktualisieren:

```bash
cd industrielle-robotik-wise25-balancing-robot
git pull
```

---

## Tutorial 3: main.py automatisch beim Booten starten (systemd)

Nach dem Testen einmalig auf dem Pi einrichten.

**1. Service-Datei erstellen**
```bash
sudo nano /etc/systemd/system/balancing-robot.service
```

Folgenden Inhalt einfügen (Pfad ggf. anpassen):
```ini
[Unit]
Description=Balancing Robot
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/industrielle-robotik-wise25-balancing-robot/main.py
WorkingDirectory=/home/pi/industrielle-robotik-wise25-balancing-robot
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

Speichern und schließen: `Ctrl+O`, `Enter`, `Ctrl+X`

**2. Service aktivieren und starten**
```bash
sudo systemctl daemon-reload
sudo systemctl enable balancing-robot
sudo systemctl start balancing-robot
```

**Nützliche Befehle**
```bash
sudo systemctl status balancing-robot   # Status prüfen
journalctl -u balancing-robot -f        # Live-Logs anzeigen
sudo systemctl stop balancing-robot     # Manuell stoppen
sudo systemctl disable balancing-robot  # Auto-Start deaktivieren
```