Continue building LadyCRM. Phase 1 (contacts, companies, works-at, a test) is DONE
and `make build` is GREEN, do NOT rewrite or break existing files. FIRST run
`ls -R cmd internal` and `make build` to see the current state. Then implement the
NEXT missing piece in this order (skip any that already exist and work). Write ONE
file (or extend ONE file) per step, then run `make build`, fix errors, then STOP --
you will be run again for the next piece.

Schema first: extend `internal/store/store.go`'s `Init()` to ALSO create these (keep
the existing tables):
- `CREATE NODE TABLE Deal (id STRING, name STRING, amount DOUBLE, stage STRING, created_at TIMESTAMP, PRIMARY KEY(id))`
- `CREATE NODE TABLE Interaction (id STRING, at TIMESTAMP, channel STRING, summary STRING, PRIMARY KEY(id))`
- `CREATE REL TABLE DealFor (FROM Deal TO Company)`
- `CREATE REL TABLE Had (FROM Contact TO Interaction)`

Then, one file per step:

1. `internal/store/deal.go`, `AddDeal(name string, amount float64, stage, companyID string) (string, error)`
   (CREATE the Deal, then MATCH deal+company and CREATE the DealFor rel), `ListDeals()`,
   and `SetStage(dealID, newStage string) error` (update stage AND log an Interaction
   "stage -> newStage" linked to... keep it simple: just update the stage for now).
2. `internal/store/interaction.go`, `AddInteraction(contactID, channel, summary string) (string, error)`
   (CREATE Interaction with `at = id.Now()`, then MATCH contact + interaction and CREATE
   the Had rel), and `Timeline(contactID string) ([]Interaction, error)` returning the
   contact's interactions ordered by `at` ascending.
3. `internal/store/path.go`, `Path(fromContactID, toCompanyID string) (string, error)`:
   a variable-length match, e.g. `MATCH (a:Contact {id:$from})-[r* SHORTEST 1..4]-(b:Company {id:$to}) RETURN ...`
   (if SHORTEST syntax errors, try `-[* 1..4]-`); return a short human-readable string
   describing whether a path exists and its length. Keep it simple and robust.
4. `cmd/crm/main.go`, wire new subcommands: `deal add --name --amount --stage --company`,
   `deal stage <id> <newstage>`, `deal list`, `interact --contact --channel --summary`,
   `timeline <contactID>`, `path --from <contactID> --to <companyID>`.

5. `internal/rlm/rlm.go`, `Solve(instruction, data string) (string, error)`: write
   `data` to a temp file, run the external command `rrlm-solve` with args
   `-i <instruction> -d @<tmpfile> --json` via `os/exec`, capture stdout, JSON-decode
   it into a struct with an `Answer string \`json:"answer"\`` field, return Answer.
   If the command fails, return the stderr as the error. (rrlm-solve is on PATH.)
6. `cmd/crm/main.go`, wire `report "<question>"`: gather a text dump of the graph
   (all contacts, companies, deals, and interactions, one per line via simple queries),
   pass it as `data` and the question as `instruction` to `rlm.Solve`, and print the
   returned answer. Also wire `import <csvfile>`: read the CSV text, ask `rlm.Solve`
   to "return a JSON array of cleaned, de-duplicated contacts with name and email"
   over the CSV, parse the JSON, and AddContact each.

IRON RULES: ONE file per step, `make build` after each, fix before stopping. Prepared
statements + `conn.Execute(stmt, map[string]any{...})` for data; exact `lbug` API
(`*lbug.PreparedStatement`, inserts use `CREATE`; verify with `go doc` if unsure).

Stop the whole task only when `make build` and `make test` are green and `report`
exists and compiles.
