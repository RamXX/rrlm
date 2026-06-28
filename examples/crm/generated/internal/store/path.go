package store

import (
	"fmt"

	lbug "github.com/LadybugDB/go-ladybug"
)

// Path returns a human-readable description of whether a path exists between a
// Contact and a Company, and its shortest length.
func Path(conn *lbug.Connection, fromContactID, toCompanyID string) (string, error) {
	// LadybugDB requires every term projected by WITH to be aliased.
	// Bind the path to a variable so length(p) gets a proper recursive relationship.
	query := `MATCH p = (a:Contact {id:$from})-[* 1..4]-(b:Company {id:$to})
	          WITH a AS a, b AS b, p AS p, length(p) AS len
	          ORDER BY len ASC
	          LIMIT 1
	          RETURN len`

	stmt, err := conn.Prepare(query)
	if err != nil {
		return "", fmt.Errorf("prepare path: %w", err)
	}
	defer stmt.Close()

	res, err := conn.Execute(stmt, map[string]any{
		"from": fromContactID,
		"to":   toCompanyID,
	})
	if err != nil {
		return "", fmt.Errorf("execute path: %w", err)
	}
	defer res.Close()

	if !res.HasNext() {
		return fmt.Sprintf("no path from contact %s to company %s", fromContactID, toCompanyID), nil
	}

	tup, err := res.Next()
	if err != nil {
		return "", fmt.Errorf("next path: %w", err)
	}
	vals, err := tup.GetAsSlice()
	tup.Close()
	if err != nil {
		return "", fmt.Errorf("get path slice: %w", err)
	}
	length := toFloat64(vals[0])
	return fmt.Sprintf("path found: contact %s <-> company %s (length %d)",
		fromContactID, toCompanyID, int(length)), nil
}
