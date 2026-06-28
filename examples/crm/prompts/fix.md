LadyCRM builds green and most commands work, but RUNNING it surfaced three bugs. Fix
them ONE AT A TIME: make the fix, run `make build`, then REPRODUCE the specific command
to confirm it works, before moving to the next. Keep all existing working code intact.

Reproduce against a scratch DB, e.g.:
  rm -rf /tmp/t.db*; ./bin/crm --db /tmp/t.db init
  C=$(./bin/crm --db /tmp/t.db contact add --name Ada --email a@x.io | tail -1)
  K=$(./bin/crm --db /tmp/t.db company add --name Acme | tail -1)
  ./bin/crm --db /tmp/t.db link works-at --contact $C --company $K --role Eng

BUG 1 — `path --from <contact> --to <company>` errors: "Parser exception: mismatched
input 'end'". internal/store/path.go uses `AS end`; `end` is a RESERVED keyword. Rename
that alias (e.g. `dst`). Confirm the variable-length match parses (try `-[* 1..4]-`).
Keep the returned string simple. Reproduce:
  ./bin/crm --db /tmp/t.db path --from $C --to $K

BUG 2 — `timeline <contact>` prints nothing although interactions were logged. In
internal/store/interaction.go check: (a) does AddInteraction actually CREATE the `Had`
relationship from the contact to the new Interaction (MATCH the contact, CREATE
`(c)-[:Had]->(i)`)? (b) does Timeline MATCH `(c:Contact {id:$id})-[:Had]->(i:Interaction)
RETURN i.at, i.channel, i.summary ORDER BY i.at`? Fix whichever is wrong. Reproduce:
  ./bin/crm --db /tmp/t.db interact --contact $C --channel email --summary "hello"
  ./bin/crm --db /tmp/t.db timeline $C

BUG 3 — `deal stage <id> <newstage>` does not persist; `deal list` shows the old stage.
SetStage in internal/store/deal.go must run `MATCH (d:Deal {id:$id}) SET d.stage=$stage`.
Reproduce:
  D=$(./bin/crm --db /tmp/t.db deal add --name X --amount 10 --stage prospect --company $K | tail -1)
  ./bin/crm --db /tmp/t.db deal stage $D won
  ./bin/crm --db /tmp/t.db deal list   # must show stage = won

IRON RULES: one fix → `make build` → reproduce that command, before the next. Stop only
when all three commands reproduce correctly.
