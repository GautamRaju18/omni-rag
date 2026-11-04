# OmniRAG sample corpus: "Project Aurora"

A small, **entirely fictional** corpus about Helios Energy Lab's sodium-ion battery program. The people, cells, numbers and products are invented for testing.

The facts are deliberately spread across files and formats, so later phases can test multi-hop and cross-document questions. For example:

- *"Which electrolyte replaced the one used in the cell that failed at 45 °C, and who owns its qualification?"* needs the CSV (which cell failed), the meeting notes (the decision) and the JSON config (the owner).
- *"Does the S1 cell's rated cycle life match what the lab measured?"* needs the HTML product page and the CSV.

| File | Format | Contents |
|---|---|---|
| `aurora_research_notes.md` | Markdown | Project overview, chemistry, results table, config snippet |
| `cell_test_results.csv` | CSV | Cycle-test results for cells AUR-C101 … AUR-C108 |
| `aurora_config.json` | JSON | Team, workstreams, test thresholds, milestones |
| `meeting_notes_2026-03-12.txt` | Plain text | Review meeting decisions and action items |
| `helios_aurora_s1.html` | HTML | Product page for the Aurora S1 cell (with nav/footer noise) |

Office, PDF, scanned and image samples are added in Phases 2b–2d.
