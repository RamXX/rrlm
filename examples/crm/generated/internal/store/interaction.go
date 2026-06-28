package store

import (
	"fmt"

	lbug "github.com/LadybugDB/go-ladybug"
	"ladycrm/internal/id"
)

// Interaction represents a row in the Interaction node table.
type Interaction struct {
	ID      string
	At      string
	Channel string
	Summary string
}

// AddInteraction inserts a new Interaction node and links it to a Contact via Had.
func AddInteraction(conn *lbug.Connection, contactID, channel, summary string) (string, error) {
	iid := id.New()
	at := id.Now()

	stmt, err := conn.Prepare("CREATE (:Interaction {id:$id, at:$at, channel:$channel, summary:$summary})")
	if err != nil {
		return "", fmt.Errorf("prepare add interaction: %w", err)
	}
	defer stmt.Close()

	_, err = conn.Execute(stmt, map[string]any{
		"id":      iid,
		"at":      at,
		"channel": channel,
		"summary": summary,
	})
	if err != nil {
		return "", fmt.Errorf("execute add interaction: %w", err)
	}

	// Link Contact -> Interaction via Had
	linkStmt, err := conn.Prepare(
		"MATCH (c:Contact {id:$cid}), (i:Interaction {id:$iid}) CREATE (c)-[:Had]->(i)",
	)
	if err != nil {
		return "", fmt.Errorf("prepare had link: %w", err)
	}
	defer linkStmt.Close()

	_, err = conn.Execute(linkStmt, map[string]any{
		"cid": contactID,
		"iid": iid,
	})
	if err != nil {
		return "", fmt.Errorf("execute had link: %w", err)
	}
	return iid, nil
}

// ListAllInteractions returns all interactions in the database.
func ListAllInteractions(conn *lbug.Connection) ([]Interaction, error) {
	query := "MATCH (i:Interaction) RETURN i.id, i.at, i.channel, i.summary ORDER BY i.at"

	stmt, err := conn.Prepare(query)
	if err != nil {
		return nil, fmt.Errorf("prepare list all interactions: %w", err)
	}
	defer stmt.Close()

	res, err := conn.Execute(stmt, nil)
	if err != nil {
		return nil, fmt.Errorf("execute list all interactions: %w", err)
	}
	defer res.Close()

	var interactions []Interaction
	for res.HasNext() {
		tup, err := res.Next()
		if err != nil {
			return nil, fmt.Errorf("next interaction: %w", err)
		}
		vals, err := tup.GetAsSlice()
		tup.Close()
		if err != nil {
			return nil, fmt.Errorf("get interaction slice: %w", err)
		}
		interactions = append(interactions, Interaction{
			ID:      toString(vals[0]),
			At:      toString(vals[1]),
			Channel: toString(vals[2]),
			Summary: toString(vals[3]),
		})
	}
	return interactions, nil
}

// Timeline returns a contact's interactions ordered by `at` ascending.
func Timeline(conn *lbug.Connection, contactID string) ([]Interaction, error) {
	query := `MATCH (c:Contact {id:$cid})-[:Had]->(i:Interaction)
	          RETURN i.id, i.at, i.channel, i.summary
	          ORDER BY i.at ASC`

	stmt, err := conn.Prepare(query)
	if err != nil {
		return nil, fmt.Errorf("prepare timeline: %w", err)
	}
	defer stmt.Close()

	res, err := conn.Execute(stmt, map[string]any{"cid": contactID})
	if err != nil {
		return nil, fmt.Errorf("execute timeline: %w", err)
	}
	defer res.Close()

	var interactions []Interaction
	for res.HasNext() {
		tup, err := res.Next()
		if err != nil {
			return nil, fmt.Errorf("next interaction: %w", err)
		}
		vals, err := tup.GetAsSlice()
		tup.Close()
		if err != nil {
			return nil, fmt.Errorf("get interaction slice: %w", err)
		}
		interactions = append(interactions, Interaction{
			ID:      toString(vals[0]),
			At:      toString(vals[1]),
			Channel: toString(vals[2]),
			Summary: toString(vals[3]),
		})
	}
	return interactions, nil
}
