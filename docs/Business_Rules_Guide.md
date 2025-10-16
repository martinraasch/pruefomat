# Business Rules für Maßnahmen-Ableitung

Das System kombiniert deterministische Regeln mit Machine-Learning. Die Regeln werden gemäß ihrer `priority` ausgewertet; bei einem Treffer liefert die Regel das Ergebnis. Andernfalls greift ein ML-Fallback mit ggf. eingeschränkten Maßnahmenklassen.

## Alle 17 Maßnahmenklassen

### Kategorie 1: Automatische Freigabe (Grüne Ampel)
1. **Rechnungsprüfung** – Ampel = Grün (1) und Betrag < 50.000 €
2. **Freigabe gemäß Kompetenzkatalog** – Ampel = Grün (1) und Betrag ≥ 50.000 €

### Kategorie 2: Nachweisanforderungen (Gelbe Ampel)
3. **Beibringung Liefer-/Leistungsnachweis (vorgelagert)**
4. **Beibringung Liefer-/Leistungsnachweis (nachgelagert)**
5. **Beibringung Auftrag/Bestellung/Vertrag (vorgelagert)**
6. **Beibringung Auftrag/Bestellung/Vertrag (nachgelagert)**
7. **telefonische Rechnungsbestätigung (vorgelagert)**
8. **telefonische Rechnungsbestätigung (nachgelagert)**

### Kategorie 3: Intensive Prüfung (Rote Ampel)
9. **telefonische Lieferbestätigung (vorgelagert)**
10. **telefonische Lieferbestätigung (nachgelagert)**
11. **schriftliche Saldenbestätigung (vorgelagert)**
12. **schriftliche Saldenbestätigung (nachgelagert)**
13. **telefonische Saldenbestätigung (nachgelagert)**
14. **schriftliche Rechnungsbestätigung beim DEB (vorgelagert)**
15. **schriftliche Rechnungsbestätigung beim DEB (nachgelagert)**
16. **nur zur Belegerfassung**

### Kategorie 4: Sonderfälle
17. **Gutschriftsverfahren** – historisches Muster (≥ 2 × „Gutschriftsverfahren“ für BUK + Debitor)

## Entscheidungslogik

```mermaid
graph TD
    A[Neue Rechnung] --> B{negativ=True?}
    B -->|Ja| Z[Bereits abgelehnt (negativ)]
    B -->|Nein| C{Ampel?}

    C -->|Grün (1)| D{Betrag?}
    D -->|< 50k| E[Rechnungsprüfung]
    D -->|≥ 50k| F[Freigabe gemäß Kompetenzkatalog]

    C -->|Gelb (2)| G[ML: Nachweis-Kategorie 3–8]
    C -->|Rot (3)| H[ML: Intensive Prüfung 9–16]

    H --> I{BUK+Debitor ≥ 2× Gutschriftsverfahren?}
    G --> I
    I -->|Ja| J[Gutschriftsverfahren]
    I -->|Nein| K[ML-Fallback]
```

## Wichtige Hinweise

### Negativ-Spalte
- **Training:** wird als Feature genutzt.
- **Prediction:** Datensätze mit `negativ=True` werden gefiltert und mit „Bereits abgelehnt (negativ)“ zurückgegeben (Score = 100 %, Quelle = `negativ_flag`).

### Betragsgrenzen
- Bei **grüner Ampel** entscheidet der Betrag: `< 50.000 €` → Rechnungsprüfung, `≥ 50.000 €` → Freigabe gemäß Kompetenzkatalog.
- Die Grenze spielt bei gelber/roter Ampel keine Rolle (ML übernimmt mit erlaubten Klassen).

### ML-Klassen-Einschränkung
- **Gelbe Ampel (2):** ML wählt ausschließlich aus den Nachweis-Maßnahmen (Klassen 3–8).
- **Rote Ampel (3):** ML wählt ausschließlich aus den intensiven Prüfmaßnahmen (Klassen 9–16).
- **Grüne Ampel (1):** komplette Entscheidungslogik über Regeln, kein ML-Einsatz.

### Historische Gutschriften
- Das System prüft BUK + Debitor-Kombinationen. Ab zwei Gutschriften greift automatisch `Gutschriftsverfahren`.
- Neue Kombinationen werden wie gewohnt über die Ampelfarbe behandelt; ML setzt die passende Maßnahme innerhalb der erlaubten Klassen.

## Regeln anpassen

- Konfigurationsdatei: `config/business_rules_massnahmen.yaml`
- Jede Regel besteht aus Name, Priority, Bedingung (`condition`), Aktion (`action`) und optionalen Feldern (`confidence`, `description`, `ml_allowed_classes`).
- Nach Änderungen Anwendung/Pipeline neu starten.
