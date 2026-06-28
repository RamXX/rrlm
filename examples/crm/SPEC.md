# LadyCRM, a graph-native CRM CLI (build specification)

You are building **LadyCRM**, a command-line CRM written in Go, backed by
**LadybugDB** (an embeddable property-graph database). The whole point of using a
graph DB is that CRM is fundamentally about **relationships that evolve over
time**, who works where (and when they moved), who introduced whom, every
interaction as a timestamped event, and how people connect us to accounts. Those
are edges, not foreign keys.

Follow this spec exactly. The build must stay green at every step
(`make build` must pass). Do not invent APIs, the LadybugDB Go API and build
recipe are given below and a working skeleton is already in place.

## Hard constraints (do not change)

- Module: `ladycrm`. Single binary `crm`.
- DB: LadybugDB via `github.com/LadybugDB/go-ladybug` (package `lbug`), already in
  `go.mod`. Build tag `system_ladybug` against the system library. The `Makefile`
  already sets the CGO flags, **always build with `make build`, never bare
  `go build`**.
- Go standard library only for everything else (flags via `flag`, CSV via
  `encoding/csv`). No web framework, no ORM, no extra deps unless already in
  `go.mod`.
- Keep functions small and files focused. Write a test alongside each package.

## LadybugDB Go API (the only API you may use)

```go
import lbug "github.com/LadybugDB/go-ladybug"

db, err := lbug.OpenDatabase(path, lbug.DefaultSystemConfig()) // on-disk
// or lbug.OpenInMemoryDatabase(lbug.DefaultSystemConfig())     // tests
defer db.Close()
conn, err := lbug.OpenConnection(db)
defer conn.Close()

res, err := conn.Query("MATCH (c:Contact) RETURN c.id, c.name")   // direct
defer res.Close()
for res.HasNext() {
    tup, _ := res.Next()
    vals, _ := tup.GetAsSlice()        // []any, column order
    // or m, _ := tup.GetAsMap()        // map[string]any by column name
    tup.Close()
}

stmt, err := conn.Prepare("CREATE (:Contact {id:$id, name:$name})")  // parameterized
defer stmt.Close()
_, err = conn.Execute(stmt, map[string]any{"id": id, "name": name})
```

Query language is Cypher-like. Schema DDL:
`CREATE NODE TABLE T(prop TYPE, ..., PRIMARY KEY(prop))` and
`CREATE REL TABLE R(FROM A TO B, prop TYPE, ...)`. Types: `STRING`, `INT64`,
`DOUBLE`, `BOOL`, `TIMESTAMP`. **Always use prepared statements with parameters
for any user/CSV-derived value** (never string-concatenate into Cypher).

## Graph schema (the model)

Node tables:
- `Contact(id STRING, name STRING, email STRING, phone STRING, title STRING, created_at TIMESTAMP, PRIMARY KEY(id))`
- `Company(id STRING, name STRING, domain STRING, industry STRING, created_at TIMESTAMP, PRIMARY KEY(id))`
- `Deal(id STRING, name STRING, amount DOUBLE, stage STRING, created_at TIMESTAMP, PRIMARY KEY(id))`
- `Interaction(id STRING, at TIMESTAMP, channel STRING, summary STRING, PRIMARY KEY(id))`

Rel tables (the temporal/relationship core):
- `WorksAt(FROM Contact TO Company, role STRING, from_date TIMESTAMP, to_date TIMESTAMP)`, role history; an open role has `to_date` unset/null.
- `Introduced(FROM Contact TO Contact, at TIMESTAMP)`, who introduced whom.
- `ParticipatedIn(FROM Contact TO Deal, role STRING)`, contact's role on a deal.
- `DealFor(FROM Deal TO Company)`, the account a deal is with.
- `Had(FROM Contact TO Interaction)`, a contact logged an interaction.
- `About(FROM Interaction TO Company)` and `AboutDeal(FROM Interaction TO Deal)`, optional subject of an interaction.

IDs are short ULIDs/uuids you generate (a helper is in the skeleton). Timestamps
are RFC3339 strings passed as parameters.

## CLI surface (`crm <group> <action> [flags]`)

Global flag `--db <path>` (default `~/.ladycrm/db`). `crm init` creates the DB and
runs all schema DDL idempotently (guard with `CREATE NODE TABLE IF NOT EXISTS` /
catch "already exists").

- `crm contact add --name --email [--phone --title]` -> prints new id
- `crm contact list` / `crm contact get <id>` / `crm contact rm <id>`
- `crm company add --name [--domain --industry]` / `company list`
- `crm deal add --name --amount --stage --company <id>` (creates `DealFor`)
- `crm deal stage <id> <new-stage>` -> updates stage AND logs an Interaction
  ("stage X -> Y") so the change is in the timeline
- `crm deal list`
- `crm link works-at --contact <id> --company <id> --role [--from --to]`
- `crm link introduced --from <contact> --to <contact>`
- `crm link deal --contact <id> --deal <id> --role`
- `crm interact --contact <id> [--company <id>] [--deal <id>] --channel --summary`
  (timestamps `at` = now unless `--at` given)
- `crm timeline <contact-id|company-id>` -> interactions in chronological order
- `crm path --from <contact-id> --to <company-id>` -> shortest relationship path
  using a variable-length Cypher match (e.g. `MATCH p = (a)-[*1..4]-(b) RETURN p`),
  printed as a readable chain ("Ada -WorksAt-> Acme")

## The data-heavy features (delegate to rrlm / rlm_solve)

Two commands analyze data too large to reason over directly. They **shell out to
`rrlm-solve`** (the RLM-first backend; `rrlm-solve` is on PATH) instead of
stuffing data into a prompt:

- `crm import <contacts.csv>`, bulk import. The CSV may have thousands of rows
  with dupes and messy fields. Pipe the CSV to `rrlm-solve` with an instruction to
  return a cleaned, de-duplicated JSON array of contacts (canonical name/email/
  company), then insert those via prepared statements. (`rrlm-solve -i "<instr>" -d @<csv> --json`,
  parse the `answer`.)
- `crm report "<question>"`, export the relevant graph (contacts, companies,
  deals, interactions) as text/JSON, then `rrlm-solve -i "<question>" -d @<export>`
  and print the answer. Natural-language analytics over the whole CRM.

Shape: build the `rrlm-solve` argv, run via `os/exec`, capture stdout, parse the
JSON result (`{"answer": ...}` with `--json`). Fail loudly if `rrlm-solve` is
absent, with a clear message.

## Suggested package layout

```
cmd/crm/main.go        # flag parsing + subcommand dispatch
internal/store/        # LadybugDB open/init + schema DDL + typed CRUD helpers
internal/model/        # Contact/Company/Deal/Interaction structs
internal/crm/          # business actions (add contact, link, timeline, path)
internal/rlm/          # rrlm-solve shell-out (import, report)
internal/id/           # id + timestamp helpers (in skeleton)
```

## Build, test, success criteria

- `make build` -> compiles `./bin/crm` (uses the system_ladybug tag + CGO flags).
- `make test` -> `go test` (use `OpenInMemoryDatabase` in tests; no network).
- Acceptance (must all work):
  1. `crm init`
  2. add 2 contacts, 1 company; `link works-at`; `interact`; `crm timeline <contact>` shows the interaction.
  3. `crm deal add` + `crm deal stage` logs a timeline entry.
  4. `crm path --from <contact> --to <company>` prints a relationship chain.
  5. `crm import sample.csv` inserts cleaned contacts (with `rrlm-solve` available).
  6. `make test` passes.

Build incrementally in this order: store+schema -> model -> contact/company CRUD ->
links -> interactions+timeline -> deals -> path -> rlm import/report -> tests.
Keep `make build` green after every step.
