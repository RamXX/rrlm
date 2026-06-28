Continue building LadyCRM Phase 1. Files already exist and `make build` is GREEN --
do NOT rewrite or break them. FIRST run `ls -R cmd internal` and `make build` to see
the current state, then implement the NEXT missing piece toward SPEC.md Phase 1, in
this order (skip any that already exist and work):

1. `internal/store/contact.go` -- a `Contact` struct, `AddContact(name, email string)
   (string, error)` (generate id with `id.New()`, set `created_at` via `id.Now()`,
   insert with a prepared statement), and `ListContacts() ([]Contact, error)`.
2. `internal/store/company.go` -- `AddCompany(name string) (string, error)` and
   `LinkWorksAt(contactID, companyID, role string) error` (MATCH both nodes, CREATE
   the WorksAt relationship).
3. `cmd/crm/main.go` -- extend the dispatch switch with subcommands: `init` (Open +
   Init), `contact add --name --email`, `contact list`, `company add --name`,
   `link works-at --contact --company --role`. Use the std `flag` package per
   subcommand; default `--db` to `~/.ladycrm/db`.
4. `internal/store/store_test.go` -- ONE test using `lbug.OpenInMemoryDatabase`:
   Init, AddContact, ListContacts, assert the contact comes back.

IRON RULES: write exactly ONE file (<= ~100 lines), then run `make build`, fix any
error before you stop. Use prepared statements + `conn.Execute(stmt, map[string]any{...})`
for all data values; use the exact `lbug` API (verify with `go doc
github.com/LadybugDB/go-ladybug.Connection` if unsure -- the type is
`*lbug.PreparedStatement`, inserts use `CREATE`). Keep `make build` green, then STOP
-- you will be run again for the next file.

Stop the whole task only when `make test` passes and `./bin/crm init`, `./bin/crm
contact add --name Ada --email ada@x.io`, and `./bin/crm contact list` all work.
