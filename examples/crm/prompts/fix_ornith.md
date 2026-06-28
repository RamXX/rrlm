LadyCRM builds green and almost every command works (init, contact, company, link,
deal add/stage/list, interact, timeline, report). RUNNING it surfaced ONE bug. Fix it,
run `make build`, then REPRODUCE the command to confirm. Keep all working code intact --
do NOT rewrite files that already work.

Reproduce against a scratch DB:
  rm -rf /tmp/t.db*; ./bin/crm --db /tmp/t.db init
  C=$(./bin/crm --db /tmp/t.db contact add --name Ada --email a@x.io | awk '{print $1}')
  K=$(./bin/crm --db /tmp/t.db company add --name Acme | awk '{print $1}')
  ./bin/crm --db /tmp/t.db link works-at --contact "$C" --company "$K" --role Eng
  ./bin/crm --db /tmp/t.db path --from "$C" --to "$K"

BUG -- `path --from <contact> --to <company>` errors at runtime:
  "prepare path: Binder exception: Expression in WITH must be aliased (use AS)."
internal/store/path.go builds a Cypher variable-length path query with a WITH clause.
LadybugDB requires every term projected by WITH to be aliased, and may not support the
`length(p)`/path projection as written. Rewrite ONLY path.go so the query parses and runs
on LadybugDB: a variable-length MATCH between the contact and the company (e.g.
`MATCH (a:Contact {id:$from})-[* 1..4]-(b:Company {id:$to})`), returning a simple
human-readable result (a hop count or a short path string). Alias every WITH term
(e.g. `WITH a AS a, b AS b`) or avoid WITH entirely. Confirm the reproduce above prints
a path/result (or a clean "no path") with NO Binder/Parser exception.

Verify the API with `go doc github.com/LadybugDB/go-ladybug.Connection` -- do not guess.

IRON RULE: one fix -> `make build` -> reproduce `path` -> stop when it runs cleanly.
