# smart-home-cat-detector

smart-home-cat-detector/
│
├── .env                       # Sensible Daten (RTSP-Passwörter, IPs) -> NIE in Git einchecken!
├── .gitignore                 # Schließt Venv, Data, Weights & Logs aus
├── README.md                  # Dokumentation (Setup-Schritte, Hardware-Setup)
├── requirements.txt           # Python-Abhängigkeiten (pip freeze / manual)
│
├── config/                    # Einstellungsdateien (YAML / JSON)
│   └── config.yaml            # Schwellenwerte, FPS, Kamera-URLs, Pfade
│
├── data/                      # Lokale Daten (KOMPLETT IN .GITIGNORE)
│   ├── raw/                   # Vom RTSP-Stream automatisch gespeicherte Rohbilder
│   ├── annotated/             # Von Label Studio / Roboflow exportiertes Dataset
│   │   ├── train/             # images/ und labels/
│   │   ├── val/               # images/ und labels/
│   │   └── data.yaml          # YOLO Dataset Config
│   └── exports/               # Pre-Labeling JSONs, Zwischenspeicher
│
├── models/                    # Gewichte & Exporte (KOMPLETT IN .GITIGNORE)
│   ├── pretrained/            # heruntergeladene yolov8m.pt, etc.
│   ├── trained/               # Deine selbst trainierten Gewichte (.pt, .pth)
│   └── exported/              # Für Raspberry Pi optimierte Modelle (.onnx, ncnn)
│
├── src/                       # Wiederverwendbarer Python-Quellcode (Modul)
│   ├── __init__.py
│   ├── capture/               # RTSP-Handling & Motion Detection
│   │   ├── __init__.py
│   │   ├── rtsp_stream.py     # Kamera-Stream-Verbindung & Reconnect-Logik
│   │   └── motion_detector.py # OpenCV MOG2 Bewegungserkennung
│   │
│   ├── dataset/               # PyTorch / Data Handling
│   │   ├── __init__.py
│   │   ├── custom_dataset.py  # PyTorch Dataset-Klasse
│   │   └── pre_labeler.py     # Logik für JSON Pre-Labeling Exporte
│   │
│   ├── models/                # Modell-Architekturen & Training
│   │   ├── __init__.py
│   │   ├── trainer.py         # PyTorch / YOLO Trainings-Loops
│   │   └── detector.py        # Wrapper-Klasse für Inferenz
│   │
│   └── utils/                 # Hilfsfunktionen
│       ├── __init__.py
│       ├── config_loader.py   # Lädt YAML / .env Dateien
│       └── visualization.py   # Zeichnet Bounding-Boxes, FPS-Counter
│
└── scripts/                   # Ausführbare Skripte (Entry Points)
    ├── collect_data.py        # Stream auslesen & Bilder speichern
    ├── generate_prelabels.py  # Erstellt JSON für Label Studio
    ├── train.py               # Startet das Training (YOLO oder PyTorch)
    ├── benchmark.py           # Vergleich Latenz / FPS
    └── deploy_pi.py           # Inferenz-Skript für den Raspberry Pi