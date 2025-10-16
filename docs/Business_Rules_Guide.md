# Business Rules für Maßnahmen-Ableitung

Das System kombiniert deterministische Business-Regeln mit einem Machine-Learning-Modell. Die Regeln werden der Reihe nach (niedrigste `priority` zuerst) ausgewertet. Greift eine Regel, liefert sie die finale Maßnahme; andernfalls fällt das System auf die ML-Vorhersage zurück.

## Regel-Hierarchie

### Priority 1 – Niedrig-Betrag & Grüne Ampel
- **Bedingung:** `Ampel = 1 (Grün)` **und** `Betrag < 50.000 €`
- **Maßnahme:** `Rechnungsprüfung`
- **Confidence:** `1.0` (100 %)
- **Beispiel:** Rechnung über 30.000 € mit Ampel Grün ⇒ automatische Rechnungsprüfung.

### Priority 2 – Historisches Gutschrift-Muster
- **Bedingung:** Für die Kombination `BUK + Debitor` wurden mindestens zwei Mal die Maßnahme `Gutschrift` vergeben.
- **Maßnahme:** `Gutschrift`
- **Confidence:** `0.9` (90 %)
- **Beispiel:** Debitor 100 im Buchungskreis A erhielt bereits drei Mal `Gutschrift` ⇒ erneut `Gutschrift`.

### Priority 3 – Negativ-Kennzeichnung
- **Bedingung:** Spalte `negativ = True`
- **Maßnahme:** `Ablehnung`
- **Confidence:** `1.0`
- **Beispiel:** Rechnung wurde vom Factoring abgelehnt (`negativ = 1`) ⇒ Maßnahme `Ablehnung`.

### Priority 999 – ML-Fallback
- **Bedingung:** Keine der vorherigen Regeln trifft zu.
- **Maßnahme:** Ergebnis des ML-Modells (`ML_PREDICTION`)
- **Confidence:** entspricht der Modellwahrscheinlichkeit.

## Regeln anpassen

Die Konfiguration der Business-Regeln befindet sich in `config/business_rules_massnahmen.yaml`. Jede Regel hat folgende Struktur:

```yaml
- name: "regel_name"
  priority: 1
  condition:
    type: "and" | "or" | "simple" | "feature_lookup" | "always"
    rules:
      - field: "Ampel"
        operator: "equals"
        value: 1
  action:
    set_field: "Massnahme_2025"
    value: "Rechnungsprüfung"
  confidence: 1.0
```

**Schritte zur Anpassung:**
1. Datei `config/business_rules_massnahmen.yaml` öffnen.
2. Regel-Blöcke ergänzen oder ändern (z. B. weitere Bedingungen, andere Aktionen).
3. Datei speichern und Anwendung neu starten bzw. Pipeline erneut trainieren.

> Tipp: Prioritäten bestimmen die Reihenfolge. Niedrigere Zahlen werden zuerst geprüft. Für Tests empfehlen wir, neue Regeln zuerst in einer Staging-Umgebung zu validieren.
