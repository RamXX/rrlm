package store

import (
	"fmt"

	lbug "github.com/LadybugDB/go-ladybug"
	"ladycrm/internal/id"
)

// Deal represents a row in the Deal node table.
type Deal struct {
	ID        string
	Name      string
	Amount    float64
	Stage     string
	CreatedAt string
}

// AddDeal inserts a new Deal node and links it to a Company via DealFor.
func AddDeal(conn *lbug.Connection, name string, amount float64, stage, companyID string) (string, error) {
	did := id.New()
	createdAt := id.Now()

	stmt, err := conn.Prepare("CREATE (:Deal {id:$id, name:$name, amount:$amount, stage:$stage, created_at:$created_at})")
	if err != nil {
		return "", fmt.Errorf("prepare add deal: %w", err)
	}
	defer stmt.Close()

	_, err = conn.Execute(stmt, map[string]any{
		"id":         did,
		"name":       name,
		"amount":     amount,
		"stage":      stage,
		"created_at": createdAt,
	})
	if err != nil {
		return "", fmt.Errorf("execute add deal: %w", err)
	}

	// Link Deal -> Company via DealFor
	linkStmt, err := conn.Prepare(
		"MATCH (d:Deal {id:$did}), (co:Company {id:$coID}) CREATE (d)-[:DealFor]->(co)",
	)
	if err != nil {
		return "", fmt.Errorf("prepare deal-for link: %w", err)
	}
	defer linkStmt.Close()

	_, err = conn.Execute(linkStmt, map[string]any{
		"did":  did,
		"coID": companyID,
	})
	if err != nil {
		return "", fmt.Errorf("execute deal-for link: %w", err)
	}
	return did, nil
}

// ListDeals returns all deals in the database.
func ListDeals(conn *lbug.Connection) ([]Deal, error) {
	query := "MATCH (d:Deal) RETURN d.id, d.name, d.amount, d.stage, d.created_at ORDER BY d.created_at"

	stmt, err := conn.Prepare(query)
	if err != nil {
		return nil, fmt.Errorf("prepare list deals: %w", err)
	}
	defer stmt.Close()

	res, err := conn.Execute(stmt, nil)
	if err != nil {
		return nil, fmt.Errorf("execute list deals: %w", err)
	}
	defer res.Close()

	var deals []Deal
	for res.HasNext() {
		tup, err := res.Next()
		if err != nil {
			return nil, fmt.Errorf("next deal: %w", err)
		}
		vals, err := tup.GetAsSlice()
		tup.Close()
		if err != nil {
			return nil, fmt.Errorf("get deal slice: %w", err)
		}
		deals = append(deals, Deal{
			ID:        toString(vals[0]),
			Name:      toString(vals[1]),
			Amount:    toFloat64(vals[2]),
			Stage:     toString(vals[3]),
			CreatedAt: toString(vals[4]),
		})
	}
	return deals, nil
}

// SetStage updates a deal's stage. Returns nil on success.
func SetStage(conn *lbug.Connection, dealID, newStage string) error {
	stmt, err := conn.Prepare("MATCH (d:Deal {id:$id}) SET d.stage=$stage")
	if err != nil {
		return fmt.Errorf("prepare set stage: %w", err)
	}
	defer stmt.Close()

	_, err = conn.Execute(stmt, map[string]any{
		"id":    dealID,
		"stage": newStage,
	})
	if err != nil {
		return fmt.Errorf("execute set stage: %w", err)
	}
	return nil
}

func toFloat64(v any) float64 {
	if v == nil {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case int32:
		return float64(x)
	default:
		return 0
	}
}
