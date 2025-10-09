# Prüfomat – Fachliche Einführung für Factoring-Teams

## 1. Zielsetzung

Der Prüfomat unterstützt Factoring-Unternehmen dabei, täglich eintreffende Rechnungen schneller vorzuselektieren. Auf Basis historischer Entscheidungen (z. B. aus den letzten Jahren) lernt das System, welche Belege mit erhöhter Wahrscheinlichkeit geprüft werden sollten. Zwei Nutzenversprechen stehen im Vordergrund:

1. **Zeitersparnis**: Das Team konzentriert sich auf die kritischsten Fälle, anstatt jede Rechnung manuell zu sichten.
2. **Transparenz**: Das System zeigt nachvollziehbar, welche Merkmale zu einer Prüfempfehlung geführt haben.

## 2. Was das System kann

- **Datenaufnahme im Excel-Format** (z. B. `Veri-Bsp.xlsx`).
- **Training auf historischen Entscheidungen** (Ampel ≥ 2 = „prüfen“).
- **Automatische Priorisierung** aktueller Belege mit einem Fraud-Score (0–100 %).
- **Erklärungen** (Feature-Importances, SHAP) zeigen, welche Faktoren das Ergebnis beeinflussen.
- **Menschliches Feedback** (✅ / ❌) wird gespeichert und kann in regelmäßiges Retraining einfließen.
- **Berichte** verraten u. a.:
  - Welche Länder/BUKs besonders häufig auffällig waren.
  - Welche Textmuster (z. B. Stichworte in „Hinweise“) gehäuft auftreten.
  - Wie präzise das Modell in der letzten Woche gearbeitet hat.

## 3. Was das System nicht kann

- **Keine automatische Entscheidung**: Es schlägt nur Prüfkandidaten vor. Die finale Freigabe/Zurückweisung liegt weiterhin beim Fachteam.
- **Keine Garantie auf Fehlerfreiheit**: Die Qualität hängt direkt von den gelieferten Daten ab (Vollständigkeit, Korrektheit, stabile Prozesse).
- **Keine eigenständige Datenbereinigung**: Offensichtliche Formatfehler (z. B. falsche Währungssymbole) müssen vorab korrigiert werden.
- **Keine Ablösung von Compliance-Regeln**: Gesetzliche Prüfschritte oder Vier-Augen-Prinzip bleiben bestehen.

## 4. Voraussetzung an die Daten

- Excel-Dateien mit Spalten wie in `Veri-Bsp.xlsx` (u. a. Land, BUK, Betrag, Ampel).
- Zeitraum: Idealerweise 2–3 Jahre historischer Daten, damit das Modell Trends erkennt.
- Zielspalte „Ampel“ mit Werten 1 (kein Befund), 2 oder 3 (prüfen).
- Optional: Freitextfelder („Hinweise“, „Maßnahme 2025“) liefern zusätzliche Muster.

## 5. Bedienungsschritte (Kurzfassung)

1. **Historische Daten aufbereiten** – Excel-Dateien prüfen, leere Spalten entfernen.
2. **Tool starten** – `python app.py` ausführen, Browser-URL öffnen.
3. **Daten laden** – Excel hochladen, Schema kontrollieren.
4. **Pipeline bauen & Baseline trainieren** – System erstellt ein Modell, berechnet Kennzahlen.
5. **Analyse nutzen** –
   - „Predictions“: Top-Prüfkandidaten mit Score.
   - „Erklärung anzeigen“: Top-5 Gründe pro Beleg.
   - „Pattern Report“: Gesamtüberblick über Muster.
6. **Batch Prediction** – neue Monats- oder Wochenlisten hochladen und Scores exportieren.
7. **Feedback geben** – Nach erfolgter Prüfung ✅ (True Positive) oder ❌ (False Positive) anklicken und optional kommentieren.
8. **Wöchentlicher Report** – Präzision und Fehlalarme kontrollieren, ggf. Retraining starten.

## 6. Umgang mit Feedback & Retraining

- Jede Feedback-Aktion landet in der SQLite-Datenbank `feedback.db` (beleg_id, Zeitstempel, Score, Kommentar).
- Bei Bedarf (z. B. >100 neue Label) die Daten exportieren und dem Training hinzufügen (`src/train_binary.py`).
- So entsteht ein kontinuierlicher Verbesserungsprozess („Human-in-the-Loop“).

## 7. Grenzen & Empfehlungen

- **Pilotbetrieb**: Start mit einem kleinen Nutzerkreis (2–3 Prüferinnen), um Prozesse anzupassen.
- **Datenqualität sichern**: Feste Verantwortlichkeiten für die Datenlieferung definieren.
- **Modell-Überwachung**: Regelmäßig Precision, Recall und Lift kontrollieren, besonders nach Prozessänderungen.
- **Dokumentation**: Entscheidungen und Feedback stets nachvollziehbar halten (auditierbar).

## 8. Ansprechpartner & Support

- **Projektleitung**: definiert, wann Retraining erfolgt und wie Feedback genutzt wird.
- **IT/Data-Team**: betreut technische Infrastruktur, führt Trainingsskripte aus.
- **Fachteam Prüfung**: bewertet Vorschläge, gibt Feedback, dokumentiert Anmerkungen.

Bei Fragen oder Änderungswünschen bitte an das interne Data-Team wenden. Gemeinsam entwickelt ihr so Stück für Stück ein praxistaugliches Frühwarnsystem für euer Factoring-Geschäft.
