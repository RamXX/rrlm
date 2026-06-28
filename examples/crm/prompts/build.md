You are building LadyCRM, a Go CLI CRM backed by LadybugDB (a graph database).
SPEC.md in this directory is the full reference (graph schema, LadybugDB Go API,
build recipe). Read it, but build only PHASE 1 below for now.

## IRON RULES (follow exactly -- violating these will fail the build)

1. ONE FILE PER STEP. Write a single file (<= ~80 lines), then immediately run
   `make build`. Fix any error BEFORE writing the next file. NEVER write two files
   in one step. NEVER write a file longer than ~100 lines -- split it.
2. ALWAYS build with `make build` (never bare `go build`). Keep it green after
   every single file.
3. A skeleton already builds: `cmd/crm/main.go` (dispatch + selftest) and
   `internal/id` (id.New(), id.Now()). Extend them; keep the build setup intact.

## LadybugDB API correctness (use exactly these -- do not guess)

- Prepared statement type is `*lbug.PreparedStatement` (NOT PreparedStmt). Verify
  with `go doc github.com/LadybugDB/go-ladybug.Connection`.
- Insert with `CREATE (:Label {prop:$p})` via a prepared statement + `conn.Execute(stmt, map[string]any{...})`.
  Use CREATE, not MERGE. Relationships: `MATCH (a:..{id:$x}),(b:..{id:$y}) CREATE (a)-[:Rel {..}]->(b)`.
- Read with `conn.Query(...)`, then `for res.HasNext() { t,_ := res.Next(); vals,_ := t.GetAsSlice(); t.Close() }`.
- Schema: `CREATE NODE TABLE T(p STRING, PRIMARY KEY(p))`, `CREATE REL TABLE R(FROM A TO B, prop TYPE)`.
  Make `init` idempotent (ignore "already exists" errors).

## PHASE 1 -- build these files in this order, `make build` after each

1. `internal/store/store.go` -- `Open(path string) (*Store, error)` (wraps Database+Connection),
   `(*Store) Close()`, `(*Store) Init() error` (runs the schema DDL for node tables
   Contact, Company and rel table WorksAt). Keep it small.
2. `internal/store/contact.go` -- `(*Store) AddContact(name, email string) (id string, err error)`
   and `(*Store) ListContacts() ([]Contact, error)` with a small `Contact` struct.
3. `internal/store/company.go` -- `(*Store) AddCompany(name string) (id string, err error)`
   and `(*Store) LinkWorksAt(contactID, companyID, role string) error`.
4. `cmd/crm/main.go` -- extend dispatch with: `init`, `contact add --name --email`,
   `contact list`, `company add --name`, `link works-at --contact --company --role`.
   Use the std `flag` package per subcommand. Default `--db` to `~/.ladycrm/db`.
5. `internal/store/store_test.go` -- ONE test using `lbug.OpenInMemoryDatabase`
   (open via a small test helper or a path-less constructor): Init, AddContact,
   ListContacts, assert the contact is returned.

## DONE when ALL pass

- `make build` green, `make test` green.
- `./bin/crm init`, then `./bin/crm contact add --name Ada --email ada@x.io`,
  then `./bin/crm contact list` shows Ada, then `./bin/crm company add --name Acme`,
  then `./bin/crm link works-at --contact <id> --company <id> --role Eng` succeed.

Stop when Phase 1 is done and green. Do NOT start deals/timeline/path/import yet.

Start now: write `internal/store/store.go` (only that file), then run `make build`.
